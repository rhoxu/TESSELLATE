import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.optimize import least_squares
from scipy.signal import find_peaks
import lightkurve as lk
from copy import deepcopy
from astropy.time import Time
import astropy.units as u
# from symfit import parameters, variables, sin, cos, Fit
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import joblib
import warnings
warnings.filterwarnings("ignore")



class get_features:
    """Calculates the 21 statistical parameters used to classify a given lightcurve"""

    def __init__(self,time=None,flux=None,mag=None,zp=None,period=None,scaled=False):
        """
        Initialise

        Parameters
        ----------
        time : array-like or None (default None)
            time axis for the lightcurve to be parameterised 
        flux : array-like or None (default None)
            flux array for the lightcurve to be parameterised 
        mag : array-like or None (default None)
            magnitude array for the lightcurve to be parameterised 
        zp : float (efault None)
            zero-point of the particular instrument used when capturing the given lightcurve. This is only required if no flux values are given
        period : array-like or None (default None)
            primary periods of the lightcurves to be parameterised (default None). If None then the periods will be calculated by the function get_periodic_features
        scaled : bool (default False)
            if true then the flux/mag array should have been scaled by the standard score: z = (x-mu)/sigma; otherwise this will be done manually
        """
        if time is None and mag is None and flux is None:
            self.names_only = True
            return 
            
        self.time = time
        if type(mag) != type(None) and type(flux) == type(None):
            try:
                self.mag = mag
                self.zp = zp
                self.flux = 10 ** (-0.4 * (self.mag - self.zp))
            except:
                raise ValueError('Magnitude lightcurve must have an accompanying zero-point value')
        elif type(mag) == type(None) and type(flux) == type(None):
            raise ValueError('You must enter lightcurve either in flux or in magnitude with accompanying zero-point')
        else:
            self.flux = flux
            self.mag = mag

        if scaled == False:
            self.flux = (self.flux - np.mean(self.flux))/np.std(self.flux)
            if type(self.mag) != type(None):
                self.mag = (self.mag - np.mean(self.mag))/np.std(self.mag)

        try:
            a = np.where((self.flux > np.percentile(self.flux,2.5)) & (self.flux < np.percentile(self.flux,97.5)))
            self.flux = self.flux[a]
        except:
            a = np.where(self.flux < np.percentile(self.flux,97.5))
            self.flux = self.flux[a]
        self.time = self.time[a]
        if type(mag) != type(None):
            self.mag = self.mag[a]

        if not isinstance(time,np.ndarray):
            time = np.array(time)
        if not isinstance(flux,np.ndarray):
            flux = np.array(flux)
        if not isinstance(mag,np.ndarray):
            mag = np.array(mag)

        self.period = period
        if type(self.period) == type(None):
            self.period = False
        self.error = np.ones(len(self.flux)) * np.std(self.flux)
        self.n_points = len(self.time)
        self.names_only = False

        self.run()
            

    def get_non_periodic_features(self):
        """
        Calulcates the non-periodic parameter used to classify the given lightcurves

        Outputs
        -------
        self.mean_abs_deviation : float
            mean value of the absolute deviations from a central point
        self.weighted_mean : float
            lightcurve mean weighted by the flux error
        self.weighted_std : float
            lightcurve standard deviation weighted by the flux error
        self.skewness : float
            skewness of the lightcurve (measure of the asymmetry of a distribution)
        self.kurtosis : float
            measurement of the peak of the flux distribution
        self.shapiro_w : float
            measure between 0 and 1 predicting whether a sample came from a normally distributed population
        self.stetson_k : float
            robust statistical parameter related to the peak of the magnitude distribution
        self.quartile31 : float
            interquartile range (IQR) of the given lightcurve
        self.mean_variance : float
            mean variance of the given lightcurve
        self.mag_95_gap : float
            normalised difference between the maximum lightcurve value and the 95th percentile
        self.hl_amp_ratio : float
            square root of the ratio between the weighted standard deviations of lightcurve datapoints below and above the mean flux value          
        """
        self.mean = np.mean(self.flux)
        self.median = np.median(self.flux)
        self.std = np.std(self.flux)
        self.mean_abs_deviation = np.median(np.abs(self.flux-np.median(self.flux)))

        self.weight = 1 / self.error
        self.weighted_sum = np.sum(self.weight)
        self.weighted_mean = np.sum(self.flux * self.weight) / self.weighted_sum
        self.weighted_std = np.sqrt(np.sum((self.flux - self.weighted_mean) ** 2 * self.weight) / self.weighted_sum)
        
        self.skewness = stats.skew(self.flux)
        self.kurtosis = stats.kurtosis(self.flux)
        self.shapiro_w = stats.shapiro(self.flux)[0]
        
        self.residual = (self.flux - self.median) / self.error
        self.stetson_k = np.sum(np.fabs(self.residual)) / np.sqrt(np.sum(self.residual ** 2)) / np.sqrt(len(self.flux))
        
        self.quartile31 = np.percentile(self.flux, 75) - np.percentile(self.flux, 25)
        
        self.diff = self.flux[1:] - self.flux[:len(self.flux) - 1]
        self.mean_variance = np.sum(self.diff ** 2) / (len(self.flux) - 1) / self.std ** 2
        
        self.cusum = np.cumsum(self.flux - self.weighted_mean) / len(self.flux) / self.weighted_std
        self.mm_cusum = np.max(self.cusum) - np.min(self.cusum)

        self.max_95_gap = 1 - np.percentile(self.flux, 95)/np.max(self.flux)
        
        index = np.where(self.flux > self.mean)
        lower_weight = self.weight[index]
        lower_weight_sum = np.sum(lower_weight)
        lower_mag = self.flux[index]
        lower_weighted_std = np.sum((lower_mag - self.mean) ** 2 * lower_weight) / lower_weight_sum

        index = np.where(self.flux <= self.mean)
        higher_weight = self.weight[index]
        higher_weight_sum = np.sum(higher_weight)
        higher_flux = self.flux[index]
        higher_weighted_std = np.sum((higher_flux - self.mean) ** 2 * higher_weight) / higher_weight_sum

        self.hl_amp_ratio = np.sqrt(lower_weighted_std / higher_weighted_std)


    def get_periodic_features(self):
        """
        Calulcates the non-periodic parameter used to classify the given lightcurves via Fourier analysis

        Outputs
        -------
        self.period : float
            peak period of the lightcurve power spectrum
        self.amplitude : float
            flux amplitude of the given lightcurve
        self.phi21 : float
            phase difference between the second and first orders of polynomials in the lightcurve Fourier series
        self.phi31 : float
            phase difference between the third and first orders of polynomials in the lightcurve Fourier series
        self.r21 : float
            amplitude ratio between the second and first orders of polynomials in the lightcurve Fourier series
        self.r31 : float
            amplitude ratio between the third and first orders of polynomials in the lightcurve Fourier series
        self.phase_mean_variance : float
            mean variance of the phase of each lightcurve data point
        self.phase_cusum : float
            phase-folded cumulative sum of the lightcurve
        self.slope_per10 : float
            10th percentile of the slope of the lihtcurve
        self.slope_per90 : float
            90th percentile of the slope of the lightcurve
        """
        if self.period == False:
            unit = u.electron / u.s
            light = lk.LightCurve(time=Time(self.time, format='mjd'),flux=self.flux*unit)#(self.f - np.nanmedian(self.f))*unit)
            self.periods = light.to_periodogram()
            
            p = deepcopy(self.periods)
            norm_p = p.power / np.nanmean(p.power)
            # norm_p[p.frequency.value < 0.05] = np.nan
            a = find_peaks(norm_p,prominence=3,distance=20,wlen=300)
            peak_power = p.power[a[0]].value
            peak_freq = p.frequency[a[0]].value
            
            ind = np.argsort(-a[1]['prominences'])
            peak_power = peak_power[ind]
            peak_freq = peak_freq[ind]
            
            freq_err = np.nanmedian(np.diff(p.frequency.value)) * 3
            
            signal_num = np.zeros_like(peak_freq,dtype=int)
            harmonic = np.zeros_like(peak_freq,dtype=int)
            counter = 1
            while (signal_num == 0).any():
                inds = np.where(signal_num == 0)[0]
                remaining = peak_freq[inds]
                r = (np.round(remaining / remaining[0], 1)) - remaining // remaining[0]
                harmonics = r <= freq_err
                signal_num[inds[harmonics]] = counter
                harmonic[inds[harmonics]] = (remaining[harmonics] // remaining[0])
                counter += 1
            
            self.periodogram_stats = {'peak_freq':peak_freq,'peak_power':peak_power,'signal_num':signal_num,'harmonic':harmonic}
            self.peak_period = float(1/peak_freq[0])
        else:
            self.peak_period = self.period


        def series(An, r):
            """
            Returns the nth order Fourier series for a given phase-folded lightcurve

            Parameters
            ----------
            An : array
                number of Fourier coefficients (2*n+1 where n is the order)
            r : array
                time array of phase-folded lightcurve to be analysed

            Returns
            -------
            sum : array
                Fourier series of order n for given phase-folded lightcurve
            """
            sum = np.zeros_like(r)
            for n, an in enumerate(An[0:An.size//2+1]):
                sum += an*np.cos(2*np.pi*n*r)
            for n, an in enumerate(An[An.size//2+1:]):
                sum += an*np.sin(2*np.pi*(n+1)*r)
            return sum


        def residual(An, r, signal):
            """
            Calculates the residual of the Fourier series of a given phase-folded lightcurve. The least squares method is applied to this function to minimise its output

            Parameters
            ----------
            An : array
                number of Fourier coefficients (2*n+1 where n is the order)
            r : array
                time array of the phase-folded lightcurve to be analysed
            signal : flux array of the phase-folded lightcurve to be analysed

            Returns
            -------
            residual : array
                residuals of a Fourier series with a particular set of coefficients. This parameter is minimised when the function is called with scipy's least squared method
            """
            return signal - series(An, r)
        

        index = np.argsort(self.time % self.peak_period)
        r = (self.time % self.peak_period)[index]
        signal = self.flux[index]

        order = 3
        Nu = 2*order+1
        An = np.arange(Nu)

        res = least_squares(residual, An, args=(r, signal))
        An = res.x

        p1 = [An[0],An[1],An[4],An[2],An[5],An[3],An[6]]
            
        self.amplitude = np.sqrt(p1[1] ** 2 + p1[2] ** 2)
        self.r21 = np.sqrt(p1[3] ** 2 + p1[4] ** 2) / self.amplitude
        self.r31 = np.sqrt(p1[5] ** 2 + p1[6] ** 2) / self.amplitude
        self.f_phase = np.arctan(-p1[1] / p1[2])
        self.phi21 = np.arctan(-p1[3] / p1[4]) - 2 * self.f_phase
        self.phi31 = np.arctan(-p1[5] / p1[6]) - 3 * self.f_phase
        
        phase_folded_date = self.time % self.peak_period
        sorted_index = np.argsort(phase_folded_date)

        folded_date = phase_folded_date[sorted_index]
        folded_flux = self.flux[sorted_index]

        diff = folded_flux[1:] - folded_flux[:len(folded_flux) - 1]
        self.phase_mean_variance = np.sum(diff * diff) / (len(folded_flux) - 1.) / self.weighted_std / self.weighted_std
        cs = np.cumsum(folded_flux - self.weighted_mean) / len(folded_flux) / self.weighted_std
        self.phase_cusum = np.max(cs) - np.min(cs)

        date_diff = folded_date[1:] - folded_date[:len(folded_date) - 1]
        flux_diff = folded_flux[1:] - folded_flux[:len(folded_flux) - 1]
        index = np.where(flux_diff != 0)
        date_diff = date_diff[index]
        flux_diff = flux_diff[index]

        slope = date_diff / flux_diff
        self.slope_per10, self.slope_per90 = np.percentile(slope, 10), np.percentile(slope, 90)     


    def gather_features(self):
        """
        Combines the various statistical parameters calculated by the functions get_non_periodic_features and get_periodic_features into a list to then be classified using classifind
        
        Outputs
        -------
        self.features : dictionary
            dictionary of 21 statistical parameters capable of classying the given lightcurve and their respective names. This is only output if this function is called through classifind.get_features

        Returns
        -------
        feature_names : list
            list of the same length as self.features containing only the parameter names. This is only returned if this function is called on its own
        """
        feature_names = ['amplitude', 'hl_amp_ratio', 'kurtosis', 'max_95_gap', 'mean_variance', 'mean_abs_deviation', 'period', 
                         'phase_cusum', 'phase_mean_variance', 'phi21', 'phi31', 'quartile31', 'r21', 'r31', 
                         'shapiro_w', 'skewness', 'slope_per10', 'slope_per90', 'stetson_k', 'weighted_std', 'weighted_mean']
        if self.names_only == False:
            feature_values = [self.amplitude, self.hl_amp_ratio, self.kurtosis, self.max_95_gap, self.mean_variance, self.mean_abs_deviation, self.peak_period,
                              self.phase_cusum, self.phase_mean_variance, self.phi21, self.phi31, self.quartile31, self.r21, self.r31, 
                              self.shapiro_w, self.skewness, self.slope_per10, self.slope_per90, self.stetson_k, self.weighted_std, self.weighted_mean]
            self.features = {}
            for i in range(0,len(feature_names)):
                self.features[feature_names[i]] = feature_values[i]
        else:
            self.features = feature_names

    
    def run(self):
        """Applies the feature obtaining process
        
        Parameters
        ----------
        return_features : bool (default True)
            if true then a call to get_features will return the dictionary of 21 statistical features used to classify the given lightcurve

        Returns
        -------
        features : dictionary
            dictionary of 21 statistical parameters capable of classying the given lightcurve and their respective names. This is only returned if return_features is True
        """
        self.get_non_periodic_features()
        self.get_periodic_features()
        features = self.gather_features()
        


class get_dataset:
    """Takes a set of flux lightcurves and creates a pandas dataframe with the statistical parameters capable of characterising them"""
    
    def __init__(self,lcs,periods=None,scaled=False):
        """
        Initialise
        
        Parameters
        ----------
        lcs : array
            flux lightcurves to obtain statistical parameters for
        periods : array-like or None (default None)
            primary periods of the lightcurves to be parameterised (default None). If None then the periods will be calculated manually
        scaled : bool (default False)
            if true then the flux array should have been scaled by the standard score: z = (x-mu)/sigma; if false then this will be done manually
        """
        self.lcs = lcs
        self.periods = periods
        self.scaled = scaled
        self.main()

        
    def build_table(self,lcs,periods=None,scaled=False):
        """
        Calculates 21 statistical features of a set of flux lightcurves using classifind.get_features then collates them into a pandas dataframe

        Parameters
        ----------
        lcs : array
            flux lightcurves to obtain statistical parameters for
        periods : array-like or None (default None)
            primary periods of the flux lightcurves (default None). If None then the periods will be calculated manually
        scaled : bool (default False)
            if true then the flux array should have been scaled by the standard score: z = (x-mu)/sigma; if false then this will be done manually

        outputs
        -------
        self.table : pandas DataFrame
            table of the names and values of 21 statistical parameters used to classify the flux lightcurves
        """
        names = get_features().gather_features()
        df = pd.DataFrame(columns=names)
        for i in range(0,len(lcs)):
            time, flux = lcs[i][:,0], lcs[i][:,1]
            try:
                period = periods[i]
            except:
                period = None
            try:
                if len(df) == 0:
                    df = pd.DataFrame([get_features(time=time,flux=flux,period=period,scaled=scaled).features])
                else:
                    try:
                        df = pd.concat([df, pd.DataFrame([get_features(time=time,flux=flux,period=period,scaled=scaled).features])], ignore_index=True)
                        df = df.sort_index()
                    except:
                        pass
            except:
                pass
        self.table = df


    def get_periods(self,lcs):
        """
        Estimates the primary periods of a set of flux lightcurves
        
        Parameters
        ----------
        lcs : array
            flux lightcurves to obtain statistical parameters for

        Outputs
        -------
        self.periods : list or float
            primary periods of the input lightcurves (list of floats if lcs has multiple lightcurves or single float value if there is only one lightcurve)    
        """
        self.lcs = lcs
        if type(self.lcs) == list:
            try:
                self.lcs = np.array(self.lcs)
            except:
                self.lcs = np.array(self.lcs,dtype=object)
        if len(self.lcs.shape) == 2:
            self.lcs = np.expand_dims(self.lcs,0)
        periods = []
        for i in range(0,len(self.lcs)):
            unit = u.electron / u.s
            light = lk.LightCurve(time=Time(self.lcs[i][:,0], format='mjd'),flux=self.lcs[i][:,1]*unit)#(self.f - np.nanmedian(self.f))*unit)
            per = light.to_periodogram()

            p = deepcopy(per)
            norm_p = p.power / np.nanmean(p.power)
            # norm_p[p.frequency.value < 0.05] = np.nan
            a = find_peaks(norm_p,prominence=3,distance=20,wlen=300)
            peak_power = p.power[a[0]].value
            peak_freq = p.frequency[a[0]].value

            ind = np.argsort(-a[1]['prominences'])
            peak_power = peak_power[ind]
            peak_freq = peak_freq[ind]

            freq_err = np.nanmedian(np.diff(p.frequency.value)) * 3

            signal_num = np.zeros_like(peak_freq,dtype=int)
            harmonic = np.zeros_like(peak_freq,dtype=int)
            counter = 1
            while (signal_num == 0).any():
                inds = np.where(signal_num == 0)[0]
                remaining = peak_freq[inds]
                r = (np.round(remaining / remaining[0], 1)) - remaining // remaining[0]
                harmonics = r <= freq_err
                signal_num[inds[harmonics]] = counter
                harmonic[inds[harmonics]] = (remaining[harmonics] // remaining[0])
                counter += 1

            self.periodogram_stats = {'peak_freq':peak_freq,'peak_power':peak_power,'signal_num':signal_num,'harmonic':harmonic}
            peak_period = float(1/peak_freq[0])
            periods.append(peak_period)
        if len(periods) == 1:
            self.periods = periods[0]
        else:
            self.periods = periods
        

    def main(self):
        """Applies the parameter table construction process"""
        if self.periods == False:
            self.get_periods(self.lcs)  
        self.build_table(lcs=self.lcs,periods=self.periods,scaled=self.scaled)


class classifind:
    """Takes an array of lightcufves and classifies them using a random forest classifier (RFC). The potential classifications, and the corresponding outputs are:
       {Eclipsing Binary : 0, Delta Scuti : 1, RR Lyrae : 2, Cepheid : 3, Long-Period Variable : 4, Non-Variable : 5}"""

    def __init__(self,lcs,periods=None,scaled=False,is_mag=False,zp=None,model='default',classes='default',train=False,summary=True,n_estimators=200,criterion='gini',max_depth=None,max_features='sqrt',max_samples=0.66):
        """
        Initialise

        Parameters
        ----------
        lcs : array-like
            lightcurves to obtain statistical parameters for
        periods : array-like or None (default None)
            primary periods of the lightcurves to be parameterised (default None). If None then the periods will be calculated manually
        scaled : bool (default False)
            if true then the flux array should have been scaled by the standard score: z = (x-mu)/sigma; if false then this will be done manually

        Options
        -------
        is_mag : bool (default False)
            if true then the lcs array should contain lightcurves in magnitude space; if false then they should be in flux space
        zp : float or None (default None)
            zero-point of the particular instrument used when capturing the given lightcurve (required if is_mag is true)
        model : sklearn.ensemble.RandomForestClassifier object or 'default' (default 'default')
            trained RFC model used to classify the lightcurves in lcs (if 'default' then the default mdoel trained on TESS lightcurves will be used)
        train : bool (default False)
            if true then the model will be trained using a collection of TESS lightcurves (only required if a non-default, untrained model is used)
        summary : bool (default True)
            if true then the model training/testing is summarised by a results table and confusion matrix (only applicable if train is true)
        n_estimators : int (default 200)
            the number of trees in the forest of the RFC (only applicable if model='default' and train=True)
        criterion : str (default 'gini')
            function to measure the quality of a split in the RFC (supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain) (only applicable if model='default' and train=True)
        max_depth : int or None (default None)
            maximum depth of the tree in the RFC (if None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples) (only applicable if model='default' and train=True)
        max_features : {'sqrt', 'log2', None}, int or float (default='sqrt')
            number of features to consider when looking for the best split in the RFC (only applicable if model='default' and train=True):
                - if int, then consider max_features features at each split
                - if float, then max_features is a fraction and max(1, int(max_features * n_features_in_)) features are considered at each split
                - if 'sqrt', then max_features=sqrt(n_features)
                - if 'log2', then max_features=log2(n_features)
                - if None, then max_features=n_features (18)
        max_samples : {int, float, None} (default 0.66)
            the number of samples to draw from lcs to train each base estimator
                - if int, then draw max_samples samples
                - if float, then draw max(round(n_samples * max_samples), 1) samples
                - if None, then draw lcs.shape[0] samples
        """
        self.lcs = lcs
        if type(self.lcs) == list:
            try:
                self.lcs = np.array(self.lcs)
            except:
                self.lcs = np.array(self.lcs,dtype=object)
        if len(self.lcs.shape) == 2:
            self.lcs = np.expand_dims(self.lcs,0)
        self.periods = periods
        self.scaled = scaled
        if is_mag == True:
            self.lcs = 10 ** (-0.4 * (self.lcs - self.zp))
            print('Lightcurves are in magnitude space')
        self.directory = os.path.dirname(os.path.abspath(__file__)) + '/rfc_files/'
        if model == 'default':
            if train == True:
                self.model = model
            else:
                self.model = joblib.load(self.directory+'RFC_model.joblib')
                pass
        else:
            self.model = model

        self.train = train
        self.summary = summary
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.classes = classes# ['Eclipsing Binary','Delta Scuti','RR Lyrae','Cepheid','Long-Period','Non-Variable']
        if self.classes == 'default':
            self.classes = ['Eclipsing Binary','Delta Scuti','RR Lyrae','Cepheid','Long-Period','Non-Variable']

        self.main(train=self.train)

    
    def train_and_test(self):
        """
        Train and test an RFC model using a collection of TESS lightcurves with the option to display the results

        Outputs
        -------
        self.model : sklearn.ensemble.RandomForestClassifier object
            trained RFC model to predict classify given lightcurves (already exists if a non-default model was used)
        self.accuracy : float
            accuracy of the trained RFC model when tested on a subset of a TESS lightcurve dataset
        """
        train_feats = np.load(self.directory+'RFC_features.npy')
        train_labels = np.load(self.directory+'RFC_labels.npy')
        X_train, X_test, y_train, y_test = train_test_split(train_feats,train_labels,test_size = 0.15)
        best_acc = 0

        for _ in range(10):
            if self.model == 'default':
                classifier = RandomForestClassifier(n_estimators=self.n_estimators,criterion=self.criterion,max_depth=self.max_depth,max_features=self.max_features)
            classifier.fit(X_train,y_train)
            y_pred = classifier.predict(X_test)
            # class_probs = classifier.predict_proba(X_test)
            acc = metrics.accuracy_score(y_test, y_pred)
            best_acc = max(best_acc,acc)

        if self.summary == True:
            _, ax = plt.subplots(1,1,figsize=(8,8))
            print(classification_report(y_test, y_pred, target_names=self.classes))
            print(f'The random forest classifier model had an overall accuracy of {100*best_acc:.2f}%')
            cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_,normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes)
            disp.plot(ax=ax,colorbar=False)
            plt.show()
        self.model, self.accuracy = classifier, acc

        return classifier, acc
    

    def predict(self,classifier,params,labels):
        """
        Predicts the stellar class of the lightcurves and calculates the probability of each lightcurve belonging to each class

        Parameters
        ----------
        classifier : sklearn.ensemble.RandomForestClassifier object
            RFC model used to classify the lightcurves represented by the statistical parameters in params
        params : pandas Dataframe
            table of statistical parameters that represent a set of lightcurves
        labels : list
            list of the names (strings) of each stellar class in the order matching the numerical prediction outputs

        Returns
        -------
        predictions : array
            predicted classes of the lightcurves represented by the parameters in params using the mapping {Eclipsing Binary : 0, Delta Scuti : 1, RR Lyrae : 2, Cepheid : 3, Long-Period Variable : 4, Non-Variable : 5}
        class_probs : array
            probability of the most likely classification for each lightcurve represented by the parameters in params
        """
        classes = {}
        class_preds = []

        for i in range(0,len(labels)):
            classes[i] = labels[i]
        for i in range(0,params.shape[0]):
            class_preds.append(classes[int(classifier.predict(np.expand_dims(params.values[0],0))[0])])

        self.class_preds = class_preds
        self.predictions = classifier.predict(params)
        self.class_probs = classifier.predict_proba(params)
    

    def main(self,train=False):
        """Applies the lightcurve classification process
        
        Parameters
        ----------
        train : bool (default False)
            if true then the model will be trained using a collection of TESS lightcurves (only required if a non-default, untrained model is used)
            
        Returns
        -------
        predictions : array
            predicted classes of the lightcurves represented by the parameters in params using the mapping {Eclipsing Binary : 0, Delta Scuti : 1, RR Lyrae : 2, Cepheid : 3, Long-Period Variable : 4, Non-Variable : 5}
        max_probs : array
            probability of most likely classification for each lightcurve represented by the parameters in params
        """
        if train:
            self.train_and_test()
        self.table = get_dataset(self.lcs,periods=self.periods,scaled=self.scaled).table

        print('\n')
        print(f'LCs: {self.lcs}')
        print('\n')

        print(f'Periods: {self.periods}')
        print('\n')

        print(f'Scaled: {self.scaled}')
        print('\n')

        # print(self.table)
        self.predict(self.model,self.table,self.classes)
        # self.classify = (self.predictions, np.max(self.class_probs(axis=1)))