# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Early stopping strategies for the evolutionary loop.

Two complementary criteria are provided:

  1. "MovingWindowEarlyStopping" — stops when the moving average of the
     validation fitness over a sliding window stops improving by more than
     a relative tolerance, for a number of consecutive generations
     ("patience").
  2. "AngularConvergenceEarlyStopping" — stops based on the geometric
     "direction" of the prediction vector rather than its magnitude: it
     tracks the angle between predictions and targets and stops once that
     angle stabilizes (small generation-to-generation change) for
     "patience" generations.

In addition to convergence, both strategies also detect degradation: a
sustained *upward* trend (positive linear-regression slope over the last
"deg_window" generations) in the metric they track — validation fitness for
the moving-window strategy, angle for the angular strategy. If that trend
holds for "patience_deg" consecutive generations, stopping is triggered
just as with convergence, and the caller still recovers the best checkpoint
via "get_best_individual" (degradation never overwrites
"best_val_individual", it only stops the loop sooner).

Both criteria always keep track of the best individual seen so far
("best_val_individual"), independently of whether they decide to trigger
an early stop, so the caller can retrieve the best checkpoint via
"get_best_individual" regardless of the outcome.

"get_early_stopping" is the factory used by calling code to instantiate
the desired strategy (or "None" if early stopping is disabled) from a
configuration dictionary.
"""

import copy
import numpy as np
import torch


def _trend_slope(values):
    """
    Slope of the least-squares linear regression line fit to a sequence of
    values, used by both strategies to detect a sustained upward
    (degrading) trend in their tracked metric.

    Parameters
    ----------
    values : sequence of float
        Values in chronological order, typically the last "deg_window"
        generations of a tracked metric (validation fitness or angle).

    Returns
    -------
    float
        Slope of the best-fit line. A positive slope means the metric is
        increasing over the window (degradation, since both tracked
        metrics — RMSE-like fitness and angle to target — are "lower is
        better"). Returns 0.0 when there are fewer than two points, since
        the slope is undefined.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x -= x.mean()
    y = np.asarray(values, dtype=float)
    return float(np.dot(x, y) / np.dot(x, x))


class EarlyStoppingBase:
    """
    Base class for early stopping strategies.

    Subclasses must implement "update" with the logic that decides,
    generation by generation, whether the stopping condition has been met.

    Parameters
    ----------
    patience : int
        Number of consecutive generations without sufficient improvement
        (as defined by the subclass) tolerated before triggering a stop.
    deg_window : int, optional
        Size of the trailing window (in generations) over which the
        degradation slope is computed. Defaults to 15.
    deg_slope_tol : float, optional
        Minimum positive slope (over "deg_window") required to count a
        generation as "degrading". Defaults to 0.1.
    patience_deg : int, optional
        Number of consecutive degrading generations tolerated before
        triggering a stop due to degradation. Defaults to 10.

    Attributes
    ----------
    patience : int
        Configured patience value.
    patience_counter : int
        Running count of consecutive generations without improvement.
    deg_counter : int
        Running count of consecutive generations classified as degrading.
    best_val_fitness : float
        Best (lowest) validation fitness observed so far.
    best_val_individual : Individual or None
        Deep copy of the individual associated with "best_val_fitness".
    stop_triggered : bool
        Whether the stopping condition has been met.
    stop_reason : str
        Human-readable description of why the stop was triggered.
    """

    def __init__(self, patience, deg_window=15, deg_slope_tol=0.1, patience_deg=10):
        self.patience = patience
        self.patience_counter = 0
        self.best_val_fitness = float('inf')
        self.best_val_individual = None
        self.stop_triggered = False
        self.stop_reason = ""
        self.deg_window = deg_window
        self.deg_slope_tol = deg_slope_tol
        self.patience_deg = patience_deg
        self.deg_counter = 0

    def update(self, generation, current_val, current_individual, y_pred=None, y_target=None):
        """
        Update internal state with the current generation's results and
        check whether the stopping condition has been met.

        Parameters
        ----------
        generation : int
            Current generation number.
        current_val : float
            Current generation's validation fitness.
        current_individual : Individual
            Individual associated with "current_val".
        y_pred : torch.Tensor or np.ndarray, optional
            Predictions on the validation set (only used by strategies that
            need them, e.g. angular convergence).
        y_target : torch.Tensor or np.ndarray, optional
            Validation targets (only used by strategies that need them).

        Returns
        -------
        bool
            "True" if the stopping condition has been triggered.

        Raises
        ------
        NotImplementedError
            Always, unless overridden by a subclass.
        """
        raise NotImplementedError("Must be implemented by a subclass")

    def get_best_individual(self):
        """
        Retrieve the best individual recorded so far.

        Returns
        -------
        Individual or None
            Deep copy of the individual with the lowest "best_val_fitness"
            seen across all calls to "update", or "None" if "update"
            has never been called.
        """
        return self.best_val_individual

    def _check_degradation(self, history):
        """
        Check whether the tracked metric shows a sustained upward
        (degrading) trend, and update the degradation patience counter
        accordingly. Shared by both subclasses: each calls this with its
        own metric history (validation fitness or angle).

        Parameters
        ----------
        history : list of float
            Full chronological history of the tracked metric. Only the
            last "deg_window" entries are used to estimate the trend.

        Returns
        -------
        bool
            "True" once the slope has stayed above "deg_slope_tol" for
            "patience_deg" consecutive generations. "False" while there is
            not yet enough history ("deg_window" points) to estimate a
            trend.
        """
        if len(history) < self.deg_window:
            return False

        slope = _trend_slope(history[-self.deg_window:])

        if slope > self.deg_slope_tol:
            self.deg_counter += 1
        else:
            self.deg_counter = 0

        return self.deg_counter >= self.patience_deg


class MovingWindowEarlyStopping(EarlyStoppingBase):
    """
    Early stopping based on the moving average of validation fitness.

    A sliding window of the last "warmup" validation fitness values is
    maintained. Once the window is full, its average is compared against
    the best moving average seen so far: if the relative improvement
    exceeds "tolerance" the counter resets, otherwise it increments.
    Stopping is triggered once the counter reaches "patience".

    Parameters
    ----------
    patience : int
        Number of consecutive non-improving windows tolerated before
        stopping.
    warmup : int
        Size of the sliding window (number of generations) used to compute
        the moving average. No stopping decision is made until the window
        is full.
    tolerance : float
        Minimum relative improvement (current vs. best moving average)
        required to reset the patience counter.
    deg_window : int, optional
        Size of the trailing window (in generations) used to estimate the
        validation-fitness degradation trend. Independent of "warmup".
        Defaults to 15.
    deg_slope_tol : float, optional
        Minimum positive slope of validation fitness (over "deg_window")
        required to count a generation as degrading. Defaults to 0.1.
    patience_deg : int, optional
        Number of consecutive degrading generations tolerated before
        stopping due to degradation. Defaults to 10.
    """

    def __init__(self, patience, warmup, tolerance, deg_window=15, deg_slope_tol=0.1, patience_deg=10):
        super().__init__(patience, deg_window=deg_window, deg_slope_tol=deg_slope_tol, patience_deg=patience_deg)
        self.warmup = warmup
        self.tolerance = tolerance
        self.val_window = []
        self.best_ma = float('inf')
        self.val_history = []

    def update(self, generation, current_val, current_individual, y_pred=None, y_target=None):
        """
        Update the sliding window and evaluate moving-average convergence.

        Parameters
        ----------
        generation : int
            Current generation number (unused directly, kept for interface
            consistency with "EarlyStoppingBase").
        current_val : float
            Current generation's validation fitness.
        current_individual : Individual
            Individual associated with "current_val".
        y_pred : torch.Tensor, optional
            Unused by this strategy.
        y_target : torch.Tensor, optional
            Unused by this strategy.

        Returns
        -------
        bool
            "True" if the moving average has stopped improving for
            "patience" consecutive windows, or if validation fitness has
            shown a sustained degrading trend for "patience_deg"
            consecutive generations.
        """
        # Track the best individual seen so far, independently of the
        # moving-average convergence check below.
        if current_val < self.best_val_fitness:
            self.best_val_fitness = current_val
            self.best_val_individual = copy.deepcopy(current_individual)

        # Maintain a fixed-size sliding window of the last "warmup" values.
        self.val_window.append(current_val)
        if len(self.val_window) > self.warmup:
            self.val_window.pop(0)

        # Only evaluate convergence once the window is fully populated.
        if len(self.val_window) == self.warmup:
            current_ma = sum(self.val_window) / self.warmup

            if self.best_ma == float('inf'):
                # First time the window is full: just initialize the
                # reference moving average, no comparison yet.
                self.best_ma = current_ma
            else:
                # Relative improvement of the current moving average over
                # the best one recorded so far (small epsilon avoids
                # division by zero when best_ma is 0).
                relative_improvement = (self.best_ma - current_ma) / (self.best_ma + 1e-8)

                if relative_improvement > self.tolerance:
                    self.best_ma = current_ma
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

            if self.patience_counter >= self.patience:
                self.stop_triggered = True
                self.stop_reason = "RMSE window convergence"

        # Degradation check: a sustained upward trend in raw validation
        # fitness over the last "deg_window" generations, independent of
        # the moving-average convergence check above. Only evaluated if
        # convergence hasn't already triggered a stop this call, mirroring
        # the priority used when both criteria could fire on the same
        # generation.
        self.val_history.append(current_val)
        if not self.stop_triggered and self._check_degradation(self.val_history):
            self.stop_triggered = True
            self.stop_reason = "RMSE window degradation"

        return self.stop_triggered


class AngularConvergenceEarlyStopping(EarlyStoppingBase):
    """
    Early stopping based on the convergence of the prediction "direction".

    Instead of monitoring fitness magnitude, this strategy tracks the angle
    (in degrees) between the prediction vector and the target vector across
    generations. The typical angle variation observed during a "warmup"
    period is used to define a tolerance ("angle_tol"); once the
    generation-to-generation angle change stays below that tolerance for
    "patience" consecutive generations, stopping is triggered.

    Parameters
    ----------
    patience : int
        Number of consecutive generations with a small angle change
        tolerated before stopping.
    warmup : int
        Number of initial generations used to estimate the typical angle
        variation ("angle_tol") before any stopping decision is made.
    tolerance : float
        Multiplier applied to the average angle change observed during
        warmup to obtain the convergence threshold "angle_tol".
    deg_window : int, optional
        Size of the trailing window (in generations) used to estimate the
        angle degradation trend. Independent of "warmup". Defaults to 15.
    deg_slope_tol : float, optional
        Minimum positive slope of the angle (over "deg_window") required
        to count a generation as degrading. Defaults to 0.1.
    patience_deg : int, optional
        Number of consecutive degrading generations tolerated before
        stopping due to degradation. Defaults to 10.
    """

    def __init__(self, patience, warmup, tolerance, deg_window=15, deg_slope_tol=0.1, patience_deg=10):
        super().__init__(patience, deg_window=deg_window, deg_slope_tol=deg_slope_tol, patience_deg=patience_deg)
        self.warmup = warmup
        self.tolerance = tolerance
        self.angle_prev_val = None
        self.angle_tol = None
        self.angle_history_val = []

    def _cumulative_angle(self, y_pred, y_target):
        """
        Compute the angle (in degrees) between the prediction and target
        vectors.

        The angle is derived from the cosine similarity between the two
        flattened vectors. A near-zero prediction vector (norm below
        "1e-10") is treated as maximally divergent (90 degrees), since
        the cosine similarity would otherwise be numerically undefined.

        Parameters
        ----------
        y_pred : torch.Tensor or np.ndarray
            Predicted values.
        y_target : torch.Tensor or np.ndarray
            Target values.

        Returns
        -------
        float
            Angle between "y_pred" and "y_target", in degrees, in the
            range [0, 180].
        """
        # Convert tensors to numpy arrays
        if torch.is_tensor(y_pred): y_pred = y_pred.detach().cpu().numpy()
        if torch.is_tensor(y_target): y_target = y_target.detach().cpu().numpy()
        
        y_pred = y_pred.flatten()
        y_target = y_target.flatten()
        
        norm_pred   = np.linalg.norm(y_pred)
        norm_target = np.linalg.norm(y_target)
        if norm_pred < 1e-10:
            return 90.0
        cos_a = np.dot(y_pred, y_target) / (norm_pred * norm_target)
        return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

    def update(self, generation, current_val, current_individual, y_pred=None, y_target=None):
        """
        Update the angle history and evaluate angular convergence.

        Parameters
        ----------
        generation : int
            Current generation number (unused directly, kept for interface
            consistency with "EarlyStoppingBase").
        current_val : float
            Current generation's validation fitness, used only for the
            best-individual checkpoint (not for the angular criterion).
        current_individual : Individual
            Individual associated with "current_val".
        y_pred : torch.Tensor or np.ndarray
            Predictions on the validation set. Required by this strategy.
        y_target : torch.Tensor or np.ndarray
            Validation targets. Required by this strategy.

        Returns
        -------
        bool
            "True" if the angle change has stayed below "angle_tol" for
            "patience" consecutive generations, or if the angle has shown
            a sustained degrading (increasing) trend for "patience_deg"
            consecutive generations.

        Raises
        ------
        ValueError
            If "y_pred" or "y_target" is not provided.
        """
        if y_pred is None or y_target is None:
            raise ValueError("AngularConvergenceEarlyStopping requires y_pred and y_target")

        # 1. Save the best historical model (RMSE checkpoint)
        if current_val < self.best_val_fitness:
            self.best_val_fitness = current_val
            self.best_val_individual = copy.deepcopy(current_individual)

        # 2. Calculate angle
        angle_val = self._cumulative_angle(y_pred, y_target)
        self.angle_history_val.append(angle_val)

        # Change in angle relative to the previous generation; 0.0 for the
        # very first generation, since there is no prior angle to compare.
        delta_angle_val = (angle_val - self.angle_prev_val) if self.angle_prev_val is not None else 0.0
        self.angle_prev_val = angle_val

        # 3. Calculate warmup tolerance
        # angle_tol is derived once, right when the warmup history becomes
        # full, from the average angle change seen during warmup. It is
        # never recomputed afterwards.
        if len(self.angle_history_val) == self.warmup:
            warmup_deltas = np.abs(np.diff(np.array(self.angle_history_val)))
            avg_delta_warmup = np.mean(warmup_deltas) if len(warmup_deltas) > 0 else 1.0
            self.angle_tol = self.tolerance * avg_delta_warmup

        # 4. Evaluate convergence
        # No convergence decision is possible until angle_tol has been set
        # (i.e. until warmup is complete).
        if self.angle_tol is not None:
            if abs(delta_angle_val) < self.angle_tol:
                self.patience_counter += 1
            else:
                self.patience_counter = 0

            if self.patience_counter >= self.patience:
                self.stop_triggered = True
                self.stop_reason = "angular convergence"

        # 5. Evaluate degradation: sustained upward trend of the angle
        # over the last "deg_window" generations, independent of the
        # previous warmup/convergence tolerance. Only evaluated if
        # convergence hasn't already triggered a stop in this same call.
        if not self.stop_triggered and self._check_degradation(self.angle_history_val):
            self.stop_triggered = True
            self.stop_reason = "angular degradation"

        return self.stop_triggered


def get_early_stopping(es_type="moving_window", **kwargs):
    """
    Factory for early stopping strategies.

    Parameters
    ----------
    es_type : str, optional
        Which strategy to build. One of "moving_window" or
        "angular" (default: "moving_window").
    **kwargs
        Configuration options forwarded to the chosen strategy:

        enable : bool
            Must be "True" for a strategy to be created at all; if
            falsy (or absent), "None" is returned regardless of
            "es_type".
        patience : int, optional
            Defaults to 10 for both strategies.
        warmup : int, optional
            Defaults to 5 for "moving_window" and 15 for "angular".
        tolerance : float, optional
            Defaults to 0.01 for "moving_window" and 0.1 for
            "angular".
        deg_window : int, optional
            Window size (generations) for degradation-slope estimation.
            Defaults to 15 for both strategies.
        deg_slope_tol : float, optional
            Minimum positive slope to count a generation as degrading.
            Defaults to 0.1 for both strategies.
        patience_deg : int, optional
            Consecutive degrading generations tolerated before stopping.
            Defaults to 10 for both strategies.

    Returns
    -------
    EarlyStoppingBase or None
        An instance of the requested strategy, or "None" if early
        stopping is disabled ("enable" is falsy).

    Raises
    ------
    ValueError
        If "es_type" is not one of the supported strategies.
    """
    if not kwargs.get('enable', False):
        return None

    if es_type == "moving_window":
        return MovingWindowEarlyStopping(
            patience=kwargs.get('patience', 10),
            warmup=kwargs.get('warmup', 5),
            tolerance=kwargs.get('tolerance', 0.01),
            deg_window=kwargs.get('deg_window', 15),
            deg_slope_tol=kwargs.get('deg_slope_tol', 0.1),
            patience_deg=kwargs.get('patience_deg', 10)
        )
    elif es_type == "angular":
        return AngularConvergenceEarlyStopping(
            patience=kwargs.get('patience', 10),
            warmup=kwargs.get('warmup', 15),     
            tolerance=kwargs.get('tolerance', 0.1),
            deg_window=kwargs.get('deg_window', 15),
            deg_slope_tol=kwargs.get('deg_slope_tol', 0.1),
            patience_deg=kwargs.get('patience_deg', 10)
        )
    else:
        raise ValueError(f"Unknown early stopping type: {es_type}")