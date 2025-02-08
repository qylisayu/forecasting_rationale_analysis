import numpy as np
import torch
from nlparam.models.abstract_model import (
    AbstractModel,
    NUM_ITERATIONS,
    NUM_CANDIDATE_PREDICATES_TO_EXPLORE,
    NUM_SAMPLES_TO_COMPUTE_PERFORMANCE,
    TEMPERATURE,
    kmeans_pp_init,
)
from nlparam import Embedder, get_validator_by_name, Proposer, DEFAULT_EMBEDDER_NAME, DEFAULT_VALIDATOR_NAME, DEFAULT_PROPOSER_NAME
from sklearn.linear_model import LogisticRegression
from tqdm import trange
import wandb
from nlparam import logger
from copy import deepcopy
from nlparam.llm.validate import validate_descriptions
from scipy.stats import pearsonr
from sklearn.metrics import f1_score



device = "cuda" if torch.cuda.is_available() else "cpu"
lsm = torch.nn.LogSoftmax(dim=1)


def lr(X, Y):
    clf = LogisticRegression(
        random_state=0,
        C=1e5,
        solver="lbfgs",
        max_iter=500,
        multi_class="multinomial",
        fit_intercept=False,
    )
    clf.fit(X, Y)
    w = clf.coef_
    probs = clf.predict_proba(X)
    loss = -np.mean(np.log(probs[np.arange(len(probs)), Y]))

    if w.shape[0] == 1:
        w = np.concatenate([np.zeros((1, w.shape[1])), w], axis=0)

    return {
        "w": w.T,
        "loss": loss,
    }


class ClassificationModel(AbstractModel):

    def __init__(
        self,
        texts,
        labels,
        scores,
        val_texts,
        val_labels,
        val_scores,
        K,
        goal,
        embedder = None,
        validator = None,
        proposer = None,
        temperature=TEMPERATURE,
        num_samples_to_compute_performance=NUM_SAMPLES_TO_COMPUTE_PERFORMANCE,
        num_iterations=NUM_ITERATIONS,
        num_candidate_predicates_to_explore=NUM_CANDIDATE_PREDICATES_TO_EXPLORE,
        reference_phi_denotation=None,
        reference_phi_predicate_strings=None,
        random_update=False,
        dummy=False,
        lr=1e-2,
    ):
        texts, reference_phi_denotation, labels, scores = AbstractModel.reorder_texts(
            texts, reference_phi_denotation=reference_phi_denotation, labels=labels, scores=scores
        )
        val_texts, _, val_labels, val_scores = AbstractModel.reorder_texts(
            texts=val_texts, labels=val_labels, scores=val_scores
        )

        # problem statement, these are cluster models' inputs
        self.texts = texts
        self.K = K
        self.goal = goal
        self.labels = labels
        self.scores = scores
        self.lr = lr

        # validation
        self.val_texts = val_texts
        self.val_labels = val_labels
        self.val_scores = val_scores

        # cluster model specific arguments
        self.commit = False
        self.absolute_correlation = True
        self.dummy = dummy

        super().__init__(
            embedder=embedder,
            validator=validator,
            proposer=proposer,
            temperature=temperature,
            num_samples_to_compute_performance=num_samples_to_compute_performance,
            num_iterations=num_iterations,
            num_candidate_predicates_to_explore=num_candidate_predicates_to_explore,
            reference_phi_denotation=reference_phi_denotation,
            reference_phi_predicate_strings=reference_phi_predicate_strings,
            random_update=random_update,
            dummy=dummy,
        )

    # the dimension of w is K x C
    def optimize_w(self, phi_denotation, w_init=None):
        results = lr(phi_denotation, self.labels)
        return results

    def optimize_tilde_phi(self, w, full_phi_denotation, optimized_ks):
        continuous_denotation_added_mask = np.zeros(
            full_phi_denotation.shape[1], dtype=np.float32
        )
        continuous_denotation_added_mask[optimized_ks] = 1.0
        continuous_denotation_added_mask = torch.tensor(
            continuous_denotation_added_mask, device=device
        )

        N, K = full_phi_denotation.shape
        N, D = self.embeddings.shape

        continuous_predicate_representation_init = kmeans_pp_init(self.embeddings, K)
        continuous_predicate_representation_parameters = torch.nn.Parameter(
            torch.tensor(continuous_predicate_representation_init).to(device)
        )
        continuous_predicate_representation_parameters.requires_grad = True
        assert continuous_predicate_representation_parameters.shape == (K, D)

        orig_denotation = torch.tensor(full_phi_denotation, device=device)
        orig_denotation[:, list(optimized_ks)] = 0.0

        w = torch.tensor(w, device=device)

        optimizer = torch.optim.Adam(
            [continuous_predicate_representation_parameters], lr=self.lr
        )

        previous_loss, initial_loss = float("inf"), None
        pbar = trange(1000)

        embeddings_torch = torch.tensor(self.embeddings, device=device)

        for step in pbar:
            normalized_continuous_predicate_parameters = (
                continuous_predicate_representation_parameters
                / continuous_predicate_representation_parameters.norm(
                    dim=1, keepdim=True
                )
            )

            continuous_denotation = torch.matmul(
                embeddings_torch, normalized_continuous_predicate_parameters.T
            )
            assert continuous_denotation.shape == (N, K)
            denotation = (
                orig_denotation
                + continuous_denotation * continuous_denotation_added_mask
            )
            logits = torch.matmul(denotation, w)  # N X C
            probs = lsm(logits)
            loss = -probs[torch.arange(len(probs)), self.labels].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss = loss.item()

            wandb.log({"tilde_phi_optimization_loss": current_loss})

            if initial_loss is None:
                initial_loss = current_loss

            if previous_loss - current_loss < 1e-4:
                break

            previous_loss = current_loss

            pbar.set_description(f"Loss: {loss.item():.4f}")

        denotation = denotation.detach().cpu().numpy()
        return denotation

    def alternatively_optimize_tilde_phi_and_w(
        self, phi_denotation, optimized_ks, num_iterations=10
    ):
        logger.debug(f"Optimizing continous predicates with idxes {optimized_ks} now")
        logger.debug(f"Optimizing w now")
        w = self.optimize_w(phi_denotation)["w"]

        pbar = range(num_iterations)
        for step in pbar:
            logger.debug(f"Optimizing tilde_phi, step {step}")
            tilde_phi_denotation = self.optimize_tilde_phi(
                w, phi_denotation, optimized_ks
            )
            logger.debug(f"Optimizing w, step {step}")
            w = self.optimize_w(tilde_phi_denotation, w_init=w)["w"]

            # Compute accuracy, loss with updated phi
            metrics = self.compute_classification_metrics(tilde_phi_denotation, w, self.labels)
            loss = self.compute_fitness_from_phi_denotation(
                tilde_phi_denotation
            )
            wandb.log({"alt_update_tilde_phi_denotation_accuracy": metrics["accuracy"], 
                       "alt_update_tilde_phi_denotation_precision": metrics["precision"],
                       "alt_update_tilde_phi_denotation_recall": metrics["recall"],
                       "alt_update_tilde_phi_denotation_f1": metrics["f1"],
                       "alt_update_tilde_phi_denotation_loss": loss, 
                       "alt_update_tilde_phi_denotation_step": step})

        return {
            "tilde_phi_denotation": tilde_phi_denotation,
            "w": w,
        }
    
    def compute_fitness_from_phi_denotation(self, phi_denotation):
        return lr(phi_denotation, self.labels)["loss"]
    
    def compute_loss(self, phi_denotation, labels):
        return lr(phi_denotation, labels)["loss"]
    
    def compute_classification_metrics(self, phi_denotation, w, labels):
        """
        Compute classification metrics: accuracy, precision, recall, F1 score, 
        and the percentage of positive predictions.

        Args:
            phi_denotation: The current (continuous) denotation of predicates.
            w: The weight matrix from training.
            labels: The ground truth labels.

        Returns:
            dict: A dictionary containing:
                - "accuracy": Accuracy of predictions.
                - "precision": Precision score.
                - "recall": Recall score.
                - "f1": F1 score.
                - "pct_positive": Percentage of predictions that are positive.
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # Generate predictions based on phi_denotation and w
        logits = phi_denotation @ w
        predictions = logits.argmax(axis=1)  # Predicted class

        # Compute accuracy
        accuracy = accuracy_score(labels, predictions)

        # Compute precision, recall, and F1 score
        precision = precision_score(labels, predictions, average="weighted")  # Weighted for imbalanced classes
        recall = recall_score(labels, predictions, average="weighted")
        f1 = f1_score(labels, predictions, average="weighted")

        # Compute percentage of predictions that are positive (class 1)
        num_positive_predictions = (predictions == 1).sum()
        total_predictions = len(predictions)
        pct_positive = (num_positive_predictions / total_predictions) * 100

        # Return results as a dictionary
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pct_positive": pct_positive
        }

    # def compute_accuracy(self, phi_denotation, w, labels):
    #     """
    #     Compute classification accuracy based on phi denotation and labels
        
    #     Args:
    #         phi_denotation: The current (continuous) denotation of predicates.
    #         w: the w from training
    #         labels: the labels
        
    #     Returns:
    #         float: The classification accuracy.
    #     """
    #     # Generate predictions based on phi_denotation and w

    #     logits = phi_denotation @ w
    #     predictions = logits.argmax(axis=1)  # Predicted class
        
    #     # Compare predictions with ground truth labels
    #     correct_predictions = (predictions == labels).sum()
    #     total_predictions = len(labels)
        
    #     return correct_predictions / total_predictions
    
    def compute_correlation(self, phi_denotation, scores, epsilon=1e-8):
        """
        Computes feature-wise correlations with cores.

        Args:
            phi_denotation (np.ndarray): Shape (N, K), denotations for each sample and feature.
            w (np.ndarray): Shape (K, N_classes), weight matrix for the logistic regression.
            scores (list or np.ndarray): Shape (N,), ground truth scores.

        Returns:
            list: A list of Pearson correlation values of shape (K,).
        """
        # Ensure scores is a NumPy array for processing
        scores = np.array(scores)

        # Add small random noise to phi_denotation and scores
        phi_denotation = phi_denotation + np.random.uniform(-epsilon, epsilon, phi_denotation.shape)

        # Initialize a list to store correlation values
        correlations = []

        # Iterate through each feature (column in phi_denotation)
        for k in range(phi_denotation.shape[1]):
            # Compute correlation between the k-th feature and scores
            correlation, _ = pearsonr(phi_denotation[:, k], scores)
            correlations.append(correlation)

        return correlations

    # def compute_f1_score_and_pct_preds_pos(self, phi_denotation, w, labels):
    #     """
    #     Compute classification F1 score and the percentage of predictions that are positive.

    #     Args:
    #         phi_denotation: The current (continuous) denotation of predicates.
    #         w: The weight matrix from training.
    #         labels: The ground truth labels.

    #     Returns:
    #         tuple: A tuple containing:
    #             - float: The F1 score.
    #             - float: The percentage of predictions that are positive.
    #     """
    #     from sklearn.metrics import f1_score

    #     # Generate predictions based on phi_denotation and w
    #     logits = phi_denotation @ w
    #     predictions = logits.argmax(axis=1)  # Predicted class
        
    #     # Compute F1 score
    #     f1 = f1_score(labels, predictions, average="weighted")  # Weighted for imbalanced classes

    #     # Compute percentage of predictions that are positive (class 1)
    #     num_positive_predictions = (predictions == 1).sum()
    #     total_predictions = len(predictions)
    #     pct_positive = (num_positive_predictions / total_predictions) * 100

    #     return f1, pct_positive


    def full_optimization_loop(self):
        init_tilde_phi = self.initialize_tilde_phi() # calls alternatively_optimize_tilde_phi_and_w()
        tilde_phi_denotation = init_tilde_phi["tilde_phi_denotation"]

        for k in range(self.K):
            continuous_denotation = tilde_phi_denotation[:, k]
            proposer_response = self.proposer.propose_descriptions(
                texts=self.texts, target=continuous_denotation, goal=self.goal
            )
            self.predicate_handler.add_predicates(proposer_response.descriptions)

        for k in range(self.K):
            best_predicate = self.predicate_handler.get_predicates_by_correlation(
                tilde_phi_denotation[:, k]
            )[0]
            self.init_predicate_strings[k] = best_predicate
            self.phi_predicate_strings[k] = best_predicate

        self.log_optimization_trajectory("init")

        train_corr = np.empty((self.K, 0)) # (K x num_iterations)
        val_corr = np.empty((self.K, 0)) # (K x num_iterations)
        for iteration in range(self.num_iterations):
            logger.debug(f"phi_predicate_denotation: {self.phi_predicate_denotation}")
            any_update_from_this_iteration = False
            train_loss = self.compute_fitness_from_phi_denotation(
                self.phi_predicate_denotation
            )
            w = self.optimize_w(self.phi_predicate_denotation)["w"]
            train_metrics = self.compute_classification_metrics(self.phi_predicate_denotation, w, self.labels)
            # current_accuracy = self.compute_accuracy(self.phi_predicate_denotation, w, self.labels)
            # current_f1, current_pct_pos = self.compute_f1_score_and_pct_preds_pos(self.phi_predicate_denotation, w, self.labels)
            train_corr_iteration = self.compute_correlation(self.phi_predicate_denotation, self.scores) # (K,)
            train_corr = np.hstack((train_corr, np.array(train_corr_iteration).reshape(-1, 1)))
            logger.debug(f"train_corr_iteration: {train_corr_iteration}")


            # Use validate_descriptions to compute the denotation matrix for validation texts
            validation_predicate_denotation = validate_descriptions(
                descriptions=self.phi_predicate_strings,  # Current predicates
                texts=self.val_texts,                    # Validation texts
                validator=self.validator,                # Validator instance
                progress_bar=True                        # Show progress bar if desired
            )
            logger.debug(f"validation_predicate_denotation: {validation_predicate_denotation}")
            val_loss = self.compute_loss(validation_predicate_denotation, self.val_labels)
            val_metrics = self.compute_classification_metrics(validation_predicate_denotation, w, self.val_labels)
            # current_val_acc = self.compute_accuracy(validation_predicate_denotation, w, self.val_labels)
            # current_val_f1, current_val_pct_pos = self.compute_f1_score_and_pct_preds_pos(validation_predicate_denotation, w, self.val_labels)
            val_corr_iteration = self.compute_correlation(validation_predicate_denotation, self.val_scores)
            val_corr = np.hstack((val_corr, np.array(val_corr_iteration).reshape(-1, 1)))
            logger.debug(f"val_corr_iteration: {val_corr_iteration}")


            wandb.log({"iteration": iteration, 
                       "training loss": train_loss, 
                       "training accuracy": train_metrics["accuracy"],
                       "training precision": train_metrics["precision"],
                       "training recall": train_metrics["recall"],
                       "training f1": train_metrics["f1"],
                       "training pct pos": train_metrics["pct_positive"],
                       "validation loss": val_loss,
                       "validation accuracy": val_metrics["accuracy"],
                       "validation precision": val_metrics["precision"],
                       "validation recall": val_metrics["recall"],
                       "validation f1": val_metrics["f1"],
                       "validation pct pos": val_metrics["pct_positive"]})
            
            # Log Train Correlations
            wandb.log({"train_corr_plot": wandb.plot.line_series(
                xs=list(range(iteration + 1)),  # X-axis: iteration
                ys=train_corr.tolist(),              # Y-axis: correlations
                keys=[f"Feature {i}" for i in range(len(train_corr_iteration))],  # Feature labels
                title="Training Correlations",
                xname="Iteration"
            )})

            # Log Validation Correlations
            wandb.log({"val_corr_plot": wandb.plot.line_series(
                xs=list(range(iteration + 1)),  # X-axis: iteration
                ys=val_corr.tolist(),              # Y-axis: correlations
                keys=[f"Feature {i}" for i in range(len(val_corr_iteration))],  # Feature labels
                title="Validation Correlations",
                xname="Iteration"
            )})



            logger.debug(
                f"Iteration {iteration}, current loss {train_loss}, current predicate strings {self.phi_predicate_strings}"
            )
            predicate_idxes_to_be_updated = self.find_idxes_by_uselessness(
                self.phi_predicate_denotation
            )
            logger.debug(
                f"Predicate idxes sorted by uselessness: {predicate_idxes_to_be_updated}"
            )

            for k in predicate_idxes_to_be_updated:
                # optimize the predicate string for k
                new_continuous_denotation = self.alternatively_optimize_tilde_phi_and_w(
                    self.phi_predicate_denotation, [k]
                )["tilde_phi_denotation"][:, k]

                new_candidate_descriptions = self.proposer.propose_descriptions(
                    texts=self.texts, target=new_continuous_denotation, goal=self.goal
                ).descriptions
                self.predicate_handler.add_predicates(new_candidate_descriptions)
                top_predicates = self.predicate_handler.get_predicates_by_correlation(
                    new_continuous_denotation
                )[: self.num_candidate_predicates_to_explore]
                logger.debug(f"Top predicates for predicate {k}: {top_predicates}")

                best_new_loss, best_new_predicate_idx = train_loss, None

                for new_predicate_idx in range(len(top_predicates)):
                    new_predicate_strings = deepcopy(self.phi_predicate_strings)
                    new_predicate_strings[k] = top_predicates[new_predicate_idx]
                    new_predicate_denotation = validate_descriptions(
                        new_predicate_strings,
                        self.texts,
                        self.validator,
                        progress_bar=True,
                    )
                    new_loss = self.compute_fitness_from_phi_denotation(
                        new_predicate_denotation
                    )
                    logger.debug(f"new_loss: {new_loss}")
                    if new_loss < best_new_loss:
                        best_new_loss, best_new_predicate_idx = (
                            new_loss,
                            new_predicate_idx,
                        )

                if best_new_predicate_idx is not None:
                    self.phi_predicate_strings[k] = top_predicates[
                        best_new_predicate_idx
                    ]
                    any_update_from_this_iteration = True

                if any_update_from_this_iteration:
                    self.log_optimization_trajectory(f"iteration_{iteration}")
                    break

            if not any_update_from_this_iteration:
                break

        return self.optimization_trajectory
