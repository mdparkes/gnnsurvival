"""
Portions of this code were adapted from survivalEVAL code by Shi-ang Qi (2021) under the MIT license:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import matplotlib.pyplot as plt
import numpy as np
import warnings
import scipy.integrate as integrate
from functools import cached_property
from scipy.integrate import trapezoid
from scipy.stats import chisquare, chi2
from typing import Callable, Optional, Union

from custom_data_types import Numeric, NumericArrayLike
from models import KaplanMeier
from utilities import check_and_convert, KaplanMeierArea
from utilities import predict_mean_survival_time, predict_median_survival_time
from utilities import predict_multi_probs_from_curve, predict_prob_from_curve


class SurvivalEvaluator:
    def __init__(
            self,
            predicted_survival_curves: NumericArrayLike,
            time_coordinates: NumericArrayLike,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
            predict_time_method: str = "Median"
    ):
        """
        Initialize the Evaluator
        param predicted_survival_curves: structured array, shape = (n_samples, n_time_points)
            Predicted survival curves for the testing samples.
        param time_coordinates: structured array, shape = (n_time_points, )
            Time coordinates for the given curves.
        param test_event_times: structured array, shape = (n_samples, )
            Actual event/censor time for the testing samples.
        param test_event_indicators: structured array, shape = (n_samples, )
            Binary indicators of censoring for the testing samples
        param train_event_times: structured array, shape = (n_train_samples, )
            Actual event/censor time for the training samples.
        param train_event_indicators: structured array, shape = (n_train_samples, )
            Binary indicators of censoring for the training samples
        """
        self._predicted_curves = check_and_convert(predicted_survival_curves)
        self._time_coordinates = check_and_convert(time_coordinates)

        test_event_times, test_event_indicators = check_and_convert(test_event_times, test_event_indicators)
        self.event_times = test_event_times
        self.event_indicators = test_event_indicators

        if (train_event_times is not None) and (train_event_indicators is not None):
            train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators

        if predict_time_method == "Median":
            self.predict_time_method = predict_median_survival_time
        elif predict_time_method == "Mean":
            self.predict_time_method = predict_mean_survival_time
        else:
            error = "Please enter one of 'Median' or 'Mean' for calculating predicted survival time."
            raise TypeError(error)

    def _error_trainset(self, method_name: str):
        if (self.train_event_times is None) or (self.train_event_indicators is None):
            raise TypeError("Train set information is missing. "
                            "Evaluator cannot perform {} evaluation.".format(method_name))

    @property
    def predicted_curves(self):
        return self._predicted_curves

    @predicted_curves.setter
    def predicted_curves(self, val):
        print("Setter called. Resetting predicted curves for this evaluator.")
        self._predicted_curves = val
        self._clear_cache()

    @property
    def time_coordinates(self):
        return self._time_coordinates

    @time_coordinates.setter
    def time_coordinates(self, val):
        print("Setter called. Resetting time coordinates for this evaluator.")
        self._time_coordinates = val
        self._clear_cache()

    @cached_property
    def predicted_event_times(self):
        return self.predict_time_from_curve(self.predict_time_method)

    def _clear_cache(self):
        # See how to clear cache in functools:
        # https://docs.python.org/3/library/functools.html#functools.cached_property
        # https://stackoverflow.com/questions/62662564/how-do-i-clear-the-cache-from-cached-property-decorator
        self.__dict__.pop('predicted_event_times', None)

    def predict_time_from_curve(
            self,
            predict_method: Callable,
    ) -> np.ndarray:
        """
        Predict survival time from survival curves.
        param predict_method: Callable
            A function that takes in a survival curve and returns a predicted survival time.
            There are two build-in methods: 'predict_median_survival_time' and 'predict_mean_survival_time'.
            'predict_median_survival_time' uses the median of the survival curve as the predicted survival time.
            'predict_mean_survival_time' uses the expected time of the survival curve as the predicted survival time.
        :return: np.ndarray
            Predicted survival time for each sample.
        """
        if (predict_method is not predict_mean_survival_time) and (predict_method is not predict_median_survival_time):
            error = "Prediction method must be 'predict_mean_survival_time' or 'predict_median_survival_time', " \
                    "got '{}' instead".format(predict_method.__name__)
            raise TypeError(error)

        predicted_times = []
        for i in range(self.predicted_curves.shape[0]):
            predicted_time = predict_method(self.predicted_curves[i, :], self.time_coordinates)
            predicted_times.append(predicted_time)
        predicted_times = np.array(predicted_times)
        return predicted_times

    def predict_probability_from_curve(
            self,
            target_time: Union[float, int, np.ndarray],
    ) -> np.ndarray:
        """
        Predict a probability of event at a given time point from a predicted curve. Each predicted curve will only
        have one corresponding probability. Note that this method is different from the
        'predict_multi_probabilities_from_curve' method, which predicts the multiple probabilities at multiple time
        points from a predicted curve.
        param target_time: float, int, or array-like, shape = (n_samples, )
            Time point(s) at which the probability of event is to be predicted. If float or int, the same time point is
            used for all samples. If array-like, each sample will have it own target time. The length of the array must
            be the same as the number of samples.
        :return: array-like, shape = (n_samples, )
            Predicted probabilities of event at the target time point(s).
        """
        if isinstance(target_time, (float, int)):
            target_time = target_time * np.ones_like(self.event_times)
        elif isinstance(target_time, np.ndarray):
            assert target_time.ndim == 1, "Target time must be a 1D array"
            assert target_time.shape[0] == self.predicted_curves.shape[0], "Target time must have the same length as " \
                                                                           "the number of samples"
        else:
            error = "Target time must be a float, int, or 1D array, got '{}' instead".format(type(target_time))
            raise TypeError(error)

        predict_probs = []
        for i in range(self.predicted_curves.shape[0]):
            predict_prob = predict_prob_from_curve(self.predicted_curves[i, :], self.time_coordinates, target_time[i])
            predict_probs.append(predict_prob)
        predict_probs = np.array(predict_probs)
        return predict_probs

    def predict_multi_probabilities_from_curve(
            self,
            target_times: np.ndarray
    ) -> np.ndarray:
        """
        Predict the probability of event at multiple time points from the predicted curve.
        param target_times: array-like, shape = (n_target_times)
            Time points at which the probability of event is to be predicted.
        :return: array-like, shape = (n_samples, n_target_times)
            Predicted probabilities of event at the target time points.
        """
        predict_probs_mat = []
        for i in range(self.predicted_curves.shape[0]):
            predict_probs = predict_multi_probs_from_curve(self.predicted_curves[i, :], self.time_coordinates,
                                                           target_times).tolist()
            predict_probs_mat.append(predict_probs)
        predict_probs_mat = np.array(predict_probs_mat)
        return predict_probs_mat

    def plot_survival_curves(
            self,
            curve_indices,
            color=None,
            x_lim: tuple = None,
            y_lim: tuple = None,
            x_label: str = 'Time',
            y_label: str = 'Survival probability'
    ):
        """Plot survival curves."""
        fig, ax = plt.subplots()
        ax.plot(self.time_coordinates, self.predicted_curves[curve_indices, :].T, color=color, label=curve_indices)
        if y_lim is None:
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylim(y_lim)

        if x_lim is not None:
            ax.set_xlim(x_lim)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        return fig, ax

    def concordance(
            self,
            ties: str = "None",
            pair_method: str = "Comparable"
    ) -> (float, float, int):
        """
        Calculate the concordance index between the predicted survival times and the true survival times.
        param ties: str, default = "None"
            A string indicating the way ties should be handled.
            Options: "None" (default), "Time", "Risk", or "All"
            "None" will throw out all ties in true survival time and all ties in predict survival times (risk scores).
            "Time" includes ties in true survival time but removes ties in predict survival times (risk scores).
            "Risk" includes ties in predict survival times (risk scores) but not in true survival time.
            "All" includes all ties.
            Note the concordance calculation is given by
            (Concordant Pairs + (Number of Ties/2))/(Concordant Pairs + Discordant Pairs + Number of Ties).
        param pair_method: str, default = "Comparable"
            A string indicating the method for constructing the pairs of samples.
            Options: "Comparable" (default) or "Margin"
            "Comparable": the pairs are constructed by comparing the predicted survival time of each sample with the
            event time of all other samples. The pairs are only constructed between samples with comparable
            event times. For example, if sample i has a censor time of 10, then the pairs are constructed by
            comparing the predicted survival time of sample i with the event time of all samples with event
            time of 10 or less.
            "Margin": the pairs are constructed between all samples. A best-guess time for the censored samples
            will be calculated and used to construct the pairs.
        :return: (float, float, int)
            The concordance index, the number of concordant pairs, and the number of total pairs.
        """
        # Choose prediction method based on the input argument
        if pair_method == "Margin" and (self.train_event_times is None or self.train_event_indicators is None):
            self._error_trainset("margin concordance")

        return concordance(self.predicted_event_times, self.event_times, self.event_indicators, self.train_event_times,
                           self.train_event_indicators, pair_method, ties)

    def brier_score(
            self,
            target_time: Optional[Union[int, float]] = None
    ) -> float:
        """
        Calculate the Brier score at a given time point from the predicted survival curve.
        param target_time: float, int, or None, default = None
            Time point at which the Brier score is to be calculated. If None, the Brier score is calculated at the
            median time of all the event/censor times from the training and test sets.
        :return: float
            The Brier score at the target time point.
        """
        self._error_trainset("Brier score (BS)")

        if target_time is None:
            target_time = np.quantile(np.concatenate((self.event_times, self.train_event_times)), 0.5)

        predict_probs = self.predict_probability_from_curve(target_time)

        return single_brier_score(predict_probs, self.event_times, self.event_indicators, self.train_event_times,
                                  self.train_event_indicators, target_time)

    def brier_score_multiple_points(
            self,
            target_times: np.ndarray
    ) -> np.ndarray:
        """
        Calculate multiple Brier scores at multiple specific times.
        param target_times: float, default: None
            The specific time points for which to estimate the Brier scores.
        :return:
            Values of multiple Brier scores.
        """
        self._error_trainset("Brier score (BS)")

        predict_probs_mat = self.predict_multi_probabilities_from_curve(target_times)

        return brier_multiple_points(predict_probs_mat, self.event_times, self.event_indicators, self.train_event_times,
                                     self.train_event_indicators, target_times)

    def integrated_brier_score(
            self,
            num_points: int = None,
            draw_figure: bool = False
    ) -> float:
        """
        Calculate the integrated Brier score (IBS) from the predicted survival curve.
        param num_points: int, default = None
            Number of points at which the Brier score is to be calculated. If None, the number of points is set to
            the number of event/censor times from the training and test sets.
        param draw_figure: bool, default = False
            Whether to draw the figure of the IBS.
        :return: float
            The integrated Brier score.
        """
        self._error_trainset("Integrated Brier Score (IBS)")

        max_target_time = np.amax(np.concatenate((self.event_times, self.train_event_times)))

        # If number of target time is not indicated, then we use the censored times obtained from test set
        if num_points is None:
            # test_censor_status = 1 - event_indicators
            censored_times = self.event_times[self.event_indicators == 0]
            sorted_censored_times = np.sort(censored_times)
            time_points = sorted_censored_times
            if time_points.size == 0:
                raise ValueError("You don't have censor data in the testset, "
                                 "please provide \"num_points\" for calculating IBS")
            else:
                time_range = np.amax(time_points) - np.amin(time_points)
        else:
            time_points = np.linspace(0, max_target_time, num_points)
            time_range = max_target_time

        # Get single brier score from multiple target times, and use trapezoidal integral to calculate ISB.
        #########################
        # Solution 1, implemented using metrics multiplication, this is geometrically faster than solution 2
        b_scores = self.brier_score_multiple_points(time_points)
        if np.isnan(b_scores).any():
            warnings.warn("Time-dependent Brier Score contains nan")
            bs_dict = {}
            for time_point, b_score in zip(time_points, b_scores):
                bs_dict[time_point] = b_score
            print("Brier scores for multiple time points are".format(bs_dict))
        integral_value = trapezoid(b_scores, time_points)
        ibs_score = integral_value / time_range
        ##########################
        # Solution 2, implemented by iteratively call single_brier_score_pycox(),
        # this solution is much slower than solution 1
        # b_scores = []
        # for i in range(len(time_points)):
        #     b_score = self.brier_score(time_points[i])
        #     b_scores.append(b_score)
        # b_scores = np.array(b_scores)
        # integral_value = trapezoid(b_scores, time_points)
        # ibs_score = integral_value / time_range

        # Draw the Brier score graph
        if draw_figure:
            plt.plot(time_points, b_scores, 'bo-')
            plt.xlabel('Time')
            plt.ylabel('Brier Score')
            plt.show()
        return ibs_score

    def l1_loss(
            self,
            method: str = "Hinge",
            weighted: bool = True,
            log_scale: bool = False
    ) -> float:
        """
        Calculate the L1 loss for the test set.
        param method: string, default: "Hinge"
            The method used to calculate the L1 loss.
            Options: "Uncensored", "Hinge" (default), "Margin", "IPCW-v1", "IPCW-v2", or "Pseudo_obs"\
        param weighted: bool, default: True
            Whether to use weighting scheme for L1 loss.
        param log_scale: boolean, default: False
            Whether to use log scale for the time axis.
        :return: float
            The L1 loss for the test set.
        """
        return l1_loss(self.predicted_event_times, self.event_times, self.event_indicators, self.train_event_times,
                       self.train_event_indicators, method, weighted, log_scale)

    def one_calibration(
            self,
            target_time: Union[float, int],
            num_bins: int = 10,
            method: str = "DN"
    ) -> (float, list, list):
        """
        Calculate the one calibration score at a given time point from the predicted survival curve.
        param target_time: float, int
            Time point at which the one calibration score is to be calculated.
        param num_bins: int, default: 10
            Number of bins used to calculate the one calibration score.
        param method: string, default: "DN"
            The method used to calculate the one calibration score.
            Options: "Uncensored", or "DN" (default)
        :return: float, list, list
            (p-value, observed probabilities, expected probabilities)
        """
        predict_probs = self.predict_probability_from_curve(target_time)
        return one_calibration(predict_probs, self.event_times, self.event_indicators, target_time, num_bins, method)

    def d_calibration(
            self,
            num_bins: int = 10
    ) -> (float, np.ndarray):
        """
        Calculate the D calibration score from the predicted survival curve.
        param num_bins: int, default: 10
            Number of bins used to calculate the D calibration score.
        :return: float, np.ndarray
            (p-value, counts in bins)
        """
        predict_probs = self.predict_probability_from_curve(self.event_times)
        return d_calibration(predict_probs, self.event_indicators, num_bins)


def single_brier_score(
        predict_probs: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        train_event_times: np.ndarray,
        train_event_indicators: np.ndarray,
        target_time: float = None
) -> float:
    """

    param predict_probs: numpy array, shape = (n_samples, )
        Estimated survival probabilities at the specific time for the testing samples.
    param event_times: numpy array, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    param event_indicators: numpy array, shape = (n_samples, )
        Binary indicators of censoring for the testing samples
    param train_event_times:numpy array, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    param train_event_indicators: numpy array, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
    param target_time: float, default: None
        The specific time point for which to estimate the Brier score.
    :return:
        Values of the brier score.
    """
    event_indicators = event_indicators.astype(bool)
    train_event_indicators = train_event_indicators.astype(bool)

    inverse_train_event_indicators = 1 - train_event_indicators
    ipc_model = KaplanMeier(train_event_times, inverse_train_event_indicators)

    ipc_pred = ipc_model.predict(event_times)
    # Catch if denominator is 0.
    ipc_pred[ipc_pred == 0] = np.inf
    # Category one calculates IPCW weight at observed time point.
    # Category one is individuals with event time lower than the time of interest and were NOT censored.
    weight_cat1 = ((event_times <= target_time) & event_indicators) / ipc_pred
    # Catch if event times goes over max training event time, i.e. predict gives NA
    weight_cat1[np.isnan(weight_cat1)] = 0
    # Category 2 is individuals whose time was greater than the time of interest (singleBrierTime)
    # contain both censored and uncensored individuals.
    weight_cat2 = (event_times > target_time) / ipc_model.predict(target_time)
    # predict returns NA if the passed-in time is greater than any of the times used to build the inverse probability
    # of censoring model.
    weight_cat2[np.isnan(weight_cat2)] = 0

    b_score = (np.square(predict_probs) * weight_cat1 + np.square(1 - predict_probs) * weight_cat2).mean()
    ###########################
    # Here we are ordering event times and then using predict with level.chaos = 1 which returns
    # predictions ordered by time.
    # This is from Haider's code in R but I feel it doesn't need to be ordered by time.
    # Refer above few lines for the justified code
    ###########################
    # order_of_times = np.argsort(event_times)
    # # Catch if event times goes over max training event time, i.e. predict gives NA
    # weight_cat1 = ((event_times[order_of_times] <= target_time) & event_indicators[order_of_times]) /\
    #               ipc_model.predict(event_times[order_of_times])
    # weight_cat1[np.isnan(weight_cat1)] = 0
    # weight_cat2 = (event_times[order_of_times] > target_time) / ipc_model.predict(target_time)
    # weight_cat2[np.isnan(weight_cat2)] = 0
    #
    # survival_curves_ordered = survival_curves[order_of_times, :]
    # predict_probs = []
    # for i in range(survival_curves_ordered.shape[0]):
    #     predict_prob = predict_prob_from_curve(survival_curves_ordered[i, :], time_coordinates,
    #                                            event_times[order_of_times][i])
    #     predict_probs.append(predict_prob)
    # predict_probs = np.array(predict_probs)
    #
    # b_score = np.mean(np.square(predict_probs) * weight_cat1 + np.square(1 - predict_probs) * weight_cat2)
    return b_score


def brier_multiple_points(
        predict_probs_mat: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        train_event_times: np.ndarray,
        train_event_indicators: np.ndarray,
        target_times: np.ndarray
) -> np.ndarray:
    """
    Calculate multiple Brier scores at multiple specific times.

    :param predict_probs_mat: structured array, shape = (n_samples, n_time_points)
        Predicted probability array (2-D) for each instances at each time point.
    :param event_times: structured array, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    :param event_indicators: structured array, shape = (n_samples, )
        Binary indicators of censoring for the testing samples
    :param train_event_times:structured array, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    :param train_event_indicators: structured array, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
    :param target_times: float, default: None
        The specific time points for which to estimate the Brier scores.
    :return:
        Values of multiple Brier scores.
    """
    inverse_train_event_indicators = 1 - train_event_indicators

    ipc_model = KaplanMeier(train_event_times, inverse_train_event_indicators)
    # sorted_test_event_times = np.argsort(event_times)

    if target_times.ndim != 1:
        error = "'time_grids' is not a one-dimensional array."
        raise TypeError(error)

    # bs_points_matrix = np.tile(event_times, (len(target_times), 1))
    target_times_mat = np.repeat(target_times.reshape(1, -1), repeats=len(event_times), axis=0)
    event_times_mat = np.repeat(event_times.reshape(-1, 1), repeats=len(target_times), axis=1)
    event_indicators_mat = np.repeat(event_indicators.reshape(-1, 1), repeats=len(target_times), axis=1)
    event_indicators_mat = event_indicators_mat.astype(bool)
    # Category one calculates IPCW weight at observed time point.
    # Category one is individuals with event time lower than the time of interest and were NOT censored.
    ipc_pred = ipc_model.predict(event_times_mat)
    # Catch if denominator is 0.
    ipc_pred[ipc_pred == 0] = np.inf
    weight_cat1 = ((event_times_mat <= target_times_mat) & event_indicators_mat) / ipc_pred
    # Catch if event times goes over max training event time, i.e. predict gives NA
    weight_cat1[np.isnan(weight_cat1)] = 0
    # Category 2 is individuals whose time was greater than the time of interest (singleBrierTime)
    # contain both censored and uncensored individuals.
    ipc_target_pred = ipc_model.predict(target_times_mat)
    # Catch if denominator is 0.
    ipc_target_pred[ipc_target_pred == 0] = np.inf
    weight_cat2 = (event_times_mat > target_times_mat) / ipc_target_pred
    # predict returns NA if the passed in time is greater than any of the times used to build
    # the inverse probability of censoring model.
    weight_cat2[np.isnan(weight_cat2)] = 0

    ipcw_square_error_mat = np.square(predict_probs_mat) * weight_cat1 + np.square(1 - predict_probs_mat) * weight_cat2
    brier_scores = np.mean(ipcw_square_error_mat, axis=0)
    return brier_scores


def concordance(
        predicted_times: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        train_event_times: Optional[np.ndarray] = None,
        train_event_indicators: Optional[np.ndarray] = None,
        pair_method: str = "Comparable",
        ties: str = "Risk"
) -> (float, float, int):
    """
    Calculate the concordance index between the predicted survival times and the true survival times.
    param predicted_times: array-like, shape = (n_samples,)
        The predicted survival times.
    param event_times: array-like, shape = (n_samples,)
        The true survival times.
    param event_indicators: array-like, shape = (n_samples,)
        The event indicators of the true survival times.
    param train_event_times: array-like, shape = (n_train_samples,)
        The true survival times of the training set.
    param train_event_indicators: array-like, shape = (n_train_samples,)
        The event indicators of the true survival times of the training set.
    param pair_method: str, optional (default="Comparable")
        A string indicating the method for constructing the pairs of samples.
        "Comparable": the pairs are constructed by comparing the predicted survival time of each sample with the
        event time of all other samples. The pairs are only constructed between samples with comparable
        event times. For example, if sample i has a censor time of 10, then the pairs are constructed by
        comparing the predicted survival time of sample i with the event time of all samples with event
        time of 10 or less.
        "Margin": the pairs are constructed between all samples. A best-guess time for the censored samples
        will be calculated and used to construct the pairs.
    param ties: str, optional (default="Risk")
        A string indicating the way ties should be handled.
        Options: "None" (default), "Time", "Risk", or "All"
        "None" will throw out all ties in true survival time and all ties in predict survival times (risk scores).
        "Time" includes ties in true survival time but removes ties in predict survival times (risk scores).
        "Risk" includes ties in predict survival times (risk scores) but not in true survival time.
        "All" includes all ties.
        Note the concordance calculation is given by
        (Concordant Pairs + (Number of Ties/2))/(Concordant Pairs + Discordant Pairs + Number of Ties).
    :return: (float, float, int)
        The concordance index, the number of concordant pairs, and the number of total pairs.
    """
    # the scikit-survival concordance function only takes risk scores to calculate.
    # So at first we should transfer the predicted time -> risk score.
    # The risk score should be higher for subjects that live shorter (i.e. lower average survival time).

    event_indicators = event_indicators.astype(bool)

    if pair_method == "Comparable":
        risks = -1 * predicted_times
        partial_weights = None
        bg_event_times = None
    elif pair_method == "Margin":
        if train_event_times is None or train_event_indicators is None:
            error = "If 'Margin' is chosen, training set information must be provided."
            raise ValueError(error)

        train_event_indicators = train_event_indicators.astype(bool)

        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        km_linear_zero = -1 / ((1 - min(km_model.survival_probabilities))/(0 - max(km_model.survival_times)))
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_model.survival_times)
        predicted_times = np.clip(predicted_times, a_max=km_linear_zero, a_min=None)
        risks = -1 * predicted_times

        censor_times = event_times[~event_indicators]
        partial_weights = np.ones_like(event_indicators, dtype=float)
        partial_weights[~event_indicators] = 1 - km_model.predict(censor_times)

        best_guesses = km_model.best_guess_revise(censor_times)
        best_guesses[censor_times > km_linear_zero] = censor_times[censor_times > km_linear_zero]

        bg_event_times = np.copy(event_times)
        bg_event_times[~event_indicators] = best_guesses
    else:
        raise TypeError("Method for calculating concordance is unrecognized.")
    # risk_ties means predicted times are the same while true times are different.
    # time_ties means true times are the same while predicted times are different.
    # cindex, concordant_pairs, discordant_pairs, risk_ties, time_ties = metrics.concordance_index_censored(
    #     event_indicators, event_times, estimate=risk)
    cindex, concordant_pairs, discordant_pairs, risk_ties, time_ties = _estimate_concordance_index(
        event_indicators, event_times, estimate=risks, bg_event_time=bg_event_times, partial_weights=partial_weights)
    if ties == "None":
        total_pairs = concordant_pairs + discordant_pairs
        cindex = concordant_pairs / total_pairs
    elif ties == "Time":
        total_pairs = concordant_pairs + discordant_pairs + time_ties
        concordant_pairs = concordant_pairs + 0.5 * time_ties
        cindex = concordant_pairs / total_pairs
    elif ties == "Risk":
        # This should be the same as original outputted cindex from above
        total_pairs = concordant_pairs + discordant_pairs + risk_ties
        concordant_pairs = concordant_pairs + 0.5 * risk_ties
        cindex = concordant_pairs / total_pairs
    elif ties == "All":
        total_pairs = concordant_pairs + discordant_pairs + risk_ties + time_ties
        concordant_pairs = concordant_pairs + 0.5 * (risk_ties + time_ties)
        cindex = concordant_pairs / total_pairs
    else:
        error = "Please enter one of 'None', 'Time', 'Risk', or 'All' for handling ties for concordance."
        raise TypeError(error)

    return cindex, concordant_pairs, total_pairs


def _estimate_concordance_index(
        event_indicator: np.ndarray,
        event_time: np.ndarray,
        estimate: np.ndarray,
        bg_event_time: np.ndarray = None,
        partial_weights: np.ndarray = None,
        tied_tol: float = 1e-8
):
    order = np.argsort(event_time, kind="stable")

    comparable, tied_time, weight = _get_comparable(event_indicator, event_time, order)

    if partial_weights is not None:
        event_indicator = np.ones_like(event_indicator)
        comparable_2, tied_time, weight = _get_comparable(event_indicator, bg_event_time, order, partial_weights)
        for ind, mask in comparable.items():
            weight[ind][mask] = 1
        comparable = comparable_2

    if len(comparable) == 0:
        raise ValueError("Data has no comparable pairs, cannot estimate concordance index.")

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]
        # w_i = partial_weights[order[ind]] # change this
        w_i = weight[ind]
        weight_i = w_i[order[mask]]

        est = estimate[order[mask]]

        assert event_i, 'got censored sample at index %d, but expected uncensored' % order[ind]

        ties = np.absolute(est - est_i) <= tied_tol
        # n_ties = ties.sum()
        n_ties = np.dot(weight_i, ties.T)
        # an event should have a higher score
        con = est < est_i
        # n_con = con[~ties].sum()
        con[ties] = False
        n_con = np.dot(weight_i, con.T)

        # numerator += w_i * n_con + 0.5 * w_i * n_ties
        # denominator += w_i * mask.sum()
        numerator += n_con + 0.5 * n_ties
        denominator += np.dot(w_i, mask.T)

        tied_risk += n_ties
        concordant += n_con
        # discordant += est.size - n_con - n_ties
        discordant += np.dot(w_i, mask.T) - n_con - n_ties

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time


def _get_comparable(event_indicator: np.ndarray, event_time: np.ndarray, order: np.ndarray,
                    partial_weights: np.ndarray = None):
    if partial_weights is None:
        partial_weights = np.ones_like(event_indicator, dtype=float)
    n_samples = len(event_time)
    tied_time = 0
    comparable = {}
    weight = {}

    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        end = i + 1
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time

        for j in range(i, end):
            if event_indicator[order[j]]:
                mask = np.zeros(n_samples, dtype=bool)
                mask[end:] = True
                # an event is comparable to censored samples at same time point
                mask[i:end] = censored_at_same_time
                comparable[j] = mask
                tied_time += censored_at_same_time.sum()
                weight[j] = partial_weights[order] * partial_weights[order[j]]
        i = end

    return comparable, tied_time, weight


def create_censor_binning(probability, num_bins) -> np.ndarray:
    """
    For censoring instance,
    b1 will be the infimum probability of the bin that contains S(c),
    for the bin of [b1, b2) which contains S(c), probability = (S(c) - b1) / S(c)
    for the rest of the bins, [b2, b3), [b3, b4), etc., probability = 1 / (B * S(c)), where B is the number of bins.
    :param probability:
        probability of the instance that will happen the event at the true event time
        based on the predicted survival curve.
    :param num_bins: number of bins
    :return: probabilities at each bin
    """
    quantile = np.linspace(1, 0, num_bins + 1)
    censor_binning = [0.0] * 10
    for i in range(num_bins):
        if probability == 1:
            censor_binning = [0.1] * 10
        elif quantile[i] > probability >= quantile[i + 1]:
            first_bin = (probability - quantile[i + 1]) / probability if probability != 0 else 1
            rest_bins = 1 / (num_bins * probability) if probability != 0 else 0
            censor_binning = [0.0] * i + [first_bin] + [rest_bins] * (num_bins - i - 1)
    # assert len(censor_binning) == 10, "censor binning should have size of 10"
    final_binning = np.array(censor_binning)
    return final_binning


def d_calibration(predict_probs, event_indicators, num_bins: int = 10):
    quantile = np.linspace(1, 0, num_bins + 1)
    censor_indicators = 1 - event_indicators

    event_probabilities = predict_probs[event_indicators.astype(bool)]
    event_position = np.digitize(event_probabilities, quantile)
    event_position[event_position == 0] = 1     # class probability==1 to the first bin

    event_binning = np.zeros([num_bins])
    for i in range(len(event_position)):
        event_binning[event_position[i] - 1] += 1

    censored_probabilities = predict_probs[censor_indicators.astype(bool)]

    censor_binning = np.zeros([num_bins])
    if len(censored_probabilities) > 0:
        for i in range(len(censored_probabilities)):
            partial_binning = create_censor_binning(censored_probabilities[i], num_bins)
            censor_binning += partial_binning

    combine_binning = event_binning + censor_binning
    _, pvalue = chisquare(combine_binning)
    return pvalue, combine_binning


def l1_loss(
        predicted_times: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        train_event_times: Optional[np.ndarray] = None,
        train_event_indicators: Optional[np.ndarray] = None,
        method: str = "Hinge",
        weighted: bool = True,
        log_scale: bool = False
) -> float:
    """
    Calculate the L1 loss for the predicted survival times.
    Parameters
    ----------
    predicted_times: np.ndarray, shape = (n_samples, )
        Predicted survival times for the testing samples
    event_times: np.ndarray, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    event_indicators: np.ndarray, shape = (n_samples, )
        Binary indicators of censoring for the testing samples
    train_event_times: np.ndarray, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    train_event_indicators: np.ndarray, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
    method: string, default: "Hinge"
        Type of l1 loss to use. Options are "Uncensored", "Hinge", "Margin", "IPCW-v1", "IPCW-v2", and "Pseudo_obs".
    weighted: boolean, default: True
        Whether to use weighting scheme for l1 loss.
    log_scale: boolean, default: False
        Whether to use log scale for the loss function.

    Returns
    -------
    Value for the calculated L1 loss.
    """
    event_indicators = event_indicators.astype(bool)
    if train_event_indicators is not None:
        train_event_indicators = train_event_indicators.astype(bool)

    if method == "Uncensored":
        if log_scale:
            scores = np.log(event_times[event_indicators]) - np.log(predicted_times[event_indicators])
        else:
            scores = event_times[event_indicators] - predicted_times[event_indicators]
        return np.abs(scores).mean()
    elif method == "Hinge":
        weights = np.ones(predicted_times.size)
        if weighted:
            if train_event_times is None or train_event_indicators is None:
                error = "If 'weighted' is True for calculating Hinge, training set values must be included."
                raise ValueError(error)
            km_model = KaplanMeierArea(train_event_times, train_event_indicators)
            censor_times = event_times[~event_indicators]
            weights[~event_indicators] = 1 - km_model.predict(censor_times)

        if log_scale:
            scores = np.log(event_times) - np.log(predicted_times)
        else:
            scores = event_times - predicted_times
        scores[~event_indicators] = np.maximum(scores[~event_indicators], 0)
        weighted_multiplier = 1 / (np.sum(event_indicators) + np.sum(weights))
        return weighted_multiplier * np.sum(np.abs(scores * weights))
    elif method == "Margin":
        if train_event_times is None or train_event_indicators is None:
            error = "If 'Margin' is chosen, training set values must be included."
            raise ValueError(error)

        # Calculate the best guess survival time given the KM curve and censoring time of that patient
        # Each best guess value has a confidence weight = 1 - KM(censoring time).
        # The earlier the patient got censored, the lower the confident weight is.
        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        km_linear_zero = -1 / ((1 - min(km_model.survival_probabilities))/(0 - max(km_model.survival_times)))
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_model.survival_times)
        # predicted_times = np.clip(predicted_times, a_max=km_linear_zero, a_min=None)

        def _km_linear_predict(time):
            slope = (1 - min(km_model.survival_probabilities)) / (0 - max(km_model.survival_times))

            # predict_prob = np.empty_like(time)
            # before_last_time_idx = time <= max(km_model.survival_times)
            # after_last_time_idx = time > max(km_model.survival_times)
            # predict_prob[before_last_time_idx] = km_model.predict(time[before_last_time_idx])
            # predict_prob[after_last_time_idx] = np.clip(1 + time[after_last_time_idx] * slope, a_min=0, a_max=None)
            if time <= max(km_model.survival_times):
                predict_prob = km_model.predict(time)
            else:
                predict_prob = max(1 + time * slope, 0)
            return predict_prob

        def _compute_best_guess(time):
            return time + integrate.quad(_km_linear_predict, time, km_linear_zero,
                                         limit=2000)[0] / km_model.predict(time)

        censor_times = event_times[~event_indicators]
        if weighted:
            weights = 1 - km_model.predict(censor_times)
        else:
            weights = np.ones(censor_times.size)
        best_guesses = km_model.best_guess_revise(censor_times)
        best_guesses[censor_times > km_linear_zero] = censor_times[censor_times > km_linear_zero]

        scores = np.empty(predicted_times.size)
        if log_scale:
            scores[event_indicators] = np.log(event_times[event_indicators]) - np.log(predicted_times[event_indicators])
            scores[~event_indicators] = weights * (np.log(best_guesses) - np.log(predicted_times[~event_indicators]))
        else:
            scores[event_indicators] = event_times[event_indicators] - predicted_times[event_indicators]
            scores[~event_indicators] = weights * (best_guesses - predicted_times[~event_indicators])
        weighted_multiplier = 1 / (np.sum(event_indicators) + np.sum(weights))
        return weighted_multiplier * np.sum(np.abs(scores))
    elif method == "IPCW-v1":
        if train_event_times is None or train_event_indicators is None:
            error = "If 'ipcw' is chosen, training set values must be included."
            raise ValueError(error)

        # Calculate the best guess survival time given the KM curve and censoring time of that patient
        # Each best guess value has a confidence weight = 1 - KM(censoring time).
        # The earlier the patient got censored, the lower the confident weight is.
        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        km_linear_zero = -1 / ((1 - min(km_model.survival_probabilities))/(0 - max(km_model.survival_times)))
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_model.survival_times)
        # predicted_times = np.clip(predicted_times, a_max=km_linear_zero, a_min=None)

        censor_times = event_times[~event_indicators]
        weights = np.ones(event_times.size)
        if weighted:
            weights[~event_indicators] = 1 - km_model.predict(censor_times)
        best_guesses = np.empty(shape=event_times.size)
        for i in range(event_times.size):
            if event_indicators[i] == 1:
                best_guesses[i] = event_times[i]
            else:
                # Numpy will throw a warning if afterward_event_times are all false. TODO: consider change the code.
                afterward_event_idx = train_event_times[train_event_indicators == 1] > event_times[i]
                best_guesses[i] = np.mean(train_event_times[train_event_indicators == 1][afterward_event_idx])
        # NaN values are generated because there are no events after the censor times
        nan_idx = np.argwhere(np.isnan(best_guesses))
        predicted_times = np.delete(predicted_times, nan_idx)
        best_guesses = np.delete(best_guesses, nan_idx)
        weights = np.delete(weights, nan_idx)
        if log_scale:
            scores = np.log(best_guesses) - np.log(predicted_times)
        else:
            scores = best_guesses - predicted_times
        weighted_multiplier = 1 / np.sum(weights)
        return weighted_multiplier * np.sum(np.abs(scores) * weights)
    elif method == "IPCW-v2":
        if train_event_times is None or train_event_indicators is None:
            error = "If 'ipcw' is chosen, training set values must be included."
            raise ValueError(error)
        # Use KM to estimate the censor distribution
        inverse_train_event_indicators = 1 - train_event_indicators

        ipc_model = KaplanMeierArea(train_event_times, inverse_train_event_indicators)
        ipc_pred = ipc_model.predict(event_times)
        # Catch if denominator is 0. This happens when the time is later than the last event time in trainset.
        ipc_pred[ipc_pred == 0] = np.inf
        if log_scale:
            scores = np.log(event_times) - np.log(predicted_times)
        else:
            scores = event_times - predicted_times
        return (np.abs(scores)[event_indicators] / ipc_pred[event_indicators]).mean()
    elif method == "Pseudo_obs":
        if train_event_times is None or train_event_indicators is None:
            error = "If 'pseudo_observation' is chosen, training set values must be included."
            raise ValueError(error)

        # Calculate the best guess survival time given the KM curve and censoring time of that patient
        # Each best guess value has a confidence weight = 1 - KM(censoring time).
        # The earlier the patient got censored, the lower the confident weight is.
        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        km_linear_zero = -1 / ((1 - min(km_model.survival_probabilities))/(0 - max(km_model.survival_times)))
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_model.survival_times)
        # predicted_times = np.clip(predicted_times, a_max=km_linear_zero, a_min=None)

        censor_times = event_times[~event_indicators]
        weights = np.ones(event_times.size)
        if weighted:
            weights[~event_indicators] = 1 - km_model.predict(censor_times)
        best_guesses = np.empty(shape=event_times.size)
        test_data_size = event_times.size
        sub_expect_time = km_model._compute_best_guess(0)
        train_data_size = train_event_times.size
        total_event_time = np.empty(shape=train_data_size + 1)
        total_event_indicator = np.empty(shape=train_data_size + 1)
        total_event_time[0:-1] = train_event_times
        total_event_indicator[0:-1] = train_event_indicators
        for i in range(test_data_size):
            if event_indicators[i] == 1:
                best_guesses[i] = event_times[i]
            else:
                total_event_time[-1] = event_times[i]
                total_event_indicator[-1] = event_indicators[i]
                total_km_model = KaplanMeierArea(total_event_time, total_event_indicator)
                total_expect_time = total_km_model._compute_best_guess(0)
                best_guesses[i] = (train_data_size + 1) * total_expect_time - train_data_size * sub_expect_time
        if log_scale:
            scores = np.log(best_guesses) - np.log(predicted_times)
        else:
            scores = best_guesses - predicted_times
        weighted_multiplier = 1 / np.sum(weights)
        return weighted_multiplier * np.sum(np.abs(scores * weights))
    elif method == "Pseudo_obs_pop":
        if train_event_times is None or train_event_indicators is None:
            error = "If 'Pseudo_obs_pop' is chosen, training set values must be included."
            raise ValueError(error)

        # Calculate the population best guess time given the KM curve.
        # The population best guess time is identical for all people.
        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        sub_expect_time = km_model._compute_best_guess(0)
        best_guesses = event_times.copy()
        best_guesses[~event_indicators] = sub_expect_time
        censor_times = event_times[~event_indicators]
        weights = np.ones(event_times.size)
        if weighted:
            weights[~event_indicators] = 1 - km_model.predict(censor_times)

        if log_scale:
            scores = np.log(best_guesses) - np.log(predicted_times)
        else:
            scores = best_guesses - predicted_times
        weighted_multiplier = 1 / np.sum(weights)
        return weighted_multiplier * np.sum(np.abs(scores * weights))
    else:
        raise ValueError("Method must be one of 'Uncensored', 'Hinge', 'Margin', 'IPCW-v1', 'IPCW-v2' "
                         "'Pseudo_obs', or 'Pseudo_obs_pop'. Got '{}' instead.".format(method))


def one_calibration(
        predictions: np.ndarray,
        event_time: np.ndarray,
        event_indicator: np.ndarray,
        target_time: Numeric,
        num_bins: int = 10,
        method: str = "DN"
) -> (float, list, list):
    """
    Compute the one calibration score for a given set of predictions and true event times.
    Parameters
    ----------
    predictions: np.ndarray
        The predicted probabilities at the time of interest.
    event_time: np.ndarray
        The true event times.
    event_indicator: np.ndarray
        The indicator of whether the event is observed or not.
    target_time: Numeric
        The time of interest.
    num_bins: int
        The number of bins to divide the predictions into.
    method: str
        The method to handle censored patients. The options are: "DN" (default), and "Uncensored".

    Returns
    -------
    score: float
        The one calibration score.
    observed_probabilities: list
        The observed probabilities in each bin.
    expected_probabilities: list
        The expected probabilities in each bin.
    """
    predictions = 1 - predictions
    sorted_idx = np.argsort(-predictions)
    sorted_predictions = predictions[sorted_idx]
    sorted_event_time = event_time[sorted_idx]
    sorted_event_indicator = event_indicator[sorted_idx]

    binned_event_time = np.array_split(sorted_event_time, num_bins)
    binned_event_indicator = np.array_split(sorted_event_indicator, num_bins)
    binned_predictions = np.array_split(sorted_predictions, num_bins)

    hl_statistics = 0
    observed_probabilities = []
    expected_probabilities = []
    for b in range(num_bins):
        # mean_prob = np.mean(binned_predictions[b])
        bin_size = len(binned_event_time[b])

        # For Uncensored method, we simply remove the censored patients,
        # for D'Agostina-Nam method, we will use 1-KM(t) as the observed probability.
        if method == "Uncensored":
            filter_idx = ~((binned_event_time[b] < target_time) & (binned_event_indicator[b] == 0))
            mean_prob = np.mean(binned_predictions[b][filter_idx])
            event_count = sum(binned_event_time[b][filter_idx] < target_time)
            event_probability = event_count / bin_size
            hl_statistics += (event_count - bin_size * mean_prob) ** 2 / (
                    bin_size * mean_prob * (1 - mean_prob))
        elif method == "DN":
            mean_prob = np.mean(binned_predictions[b])
            km_model = KaplanMeier(binned_event_time[b], binned_event_indicator[b])
            event_probability = 1 - km_model.predict(target_time)
            hl_statistics += (bin_size * event_probability - bin_size * mean_prob) ** 2 / (bin_size * mean_prob * (1 - mean_prob))
        else:
            error = "Please enter one of 'Uncensored','DN' for method."
            raise TypeError(error)
        observed_probabilities.append(event_probability)
        expected_probabilities.append(mean_prob)

    degree_of_freedom = num_bins - 1 if (num_bins <= 15 and method == "DN") else num_bins - 2
    p_value = 1 - chi2.cdf(hl_statistics, degree_of_freedom)

    return p_value, observed_probabilities, expected_probabilities

