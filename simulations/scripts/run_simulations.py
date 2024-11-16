import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from statsmodels.tsa.stattools import acf as autocorr
from scipy.optimize import curve_fit
from scipy.stats import zscore
import simulations.src.sim_params as sim_params
from packages import connect_to_dj



t = np.arange(sim_params.t_start, sim_params.t_end, sim_params.dt)
h = sim_params.Ca_trans_amp * (1 - np.exp(-t/(sim_params.tau_on))) * (np.exp(-t/(sim_params.tau_off)))


VM = connect_to_dj.get_virtual_modules()

subject_idx = 46
sesssion_num = 2

# Get velocity signal
keys = (VM['subject'].Subject & 'user_netid="pss3570"').fetch()
void_subject = pd.DataFrame((VM['behavior'].IterationData & keys[subject_idx] & 'task="IntoTheVoid"').fetch())


x_vel = void_subject.velocity[sesssion_num][:,0]
y_vel = void_subject.velocity[sesssion_num][:,1]
subject_vel = np.sqrt((x_vel ** 2) + (y_vel ** 2))

new_times = void_subject.timestamps[sesssion_num]
idx = np.where(subject_vel != 0)
vel_interp = np.interp(new_times, new_times[idx].reshape(-1), subject_vel[idx])

vel_interp = np.stack((vel_interp.reshape(-1), new_times.reshape(-1)), axis=0).T



real_dt = 0.1 # time interval, in seconnds, of the actual experimental sampling resolution
intrinsicRate = 50 # constant firing rate (spikes/s), for homogenous poisson process
cutRise = 50
renewalFactor = 4

negShift  = -2/real_dt
shiftStep = 0.5/real_dt
posShift  = 2/real_dt
halfShiftRange = (posShift - negShift)/2

signal = []

nlags = len(np.arange(sim_params.t_start, sim_params.t_end, real_dt)) - 2 - cutRise





data = []
SIM = []
maximumIteration = 10

for trainN in np.arange(maximumIteration):
    
    #offsetTime = round(np.rand(1)*(vel_interp(len(vel_interp)-1,2)-sim_params.t_end-2)) # get a random starting point (in seconds) to pull from behavioral velocity vector
    offsetTime = 150
    if offsetTime < 3:
        offsetTime = 3
    
    velCutIDXstart = np.where(np.abs(vel_interp[:,1]-offsetTime) == np.min(np.abs(vel_interp[:,1]-offsetTime)))[0][0]
    velCutIDXend = np.where(np.abs(vel_interp[:,1]-(sim_params.t_end+offsetTime)) == np.min(np.abs(vel_interp[:,1]-(sim_params.t_end+offsetTime))))[0][0]
    VelSnip = vel_interp[velCutIDXstart:velCutIDXend+5,:] # add 5 to the end of the index to give the interpolation a bit of room at the end
    
    VelSnipFast = np.interp(t, VelSnip[:,1]-VelSnip[0,1], VelSnip[:,0])

    # Independent and noise signals
    independentSignal = 5 * np.sin(np.linspace(0, 16 * np.pi, len(VelSnipFast)))
    noiseSignal = np.random.rand(len(VelSnipFast))

    # Intrinsic Rate Matrix
    intrinsicRateMatrix = [50]
    #SIM = {}

    for intrinsicRate in intrinsicRateMatrix:
    
        noiseModExtentMatrix = [0]
        for noiseModExtentIDX, noiseModExtent in enumerate(noiseModExtentMatrix):

            velocityModulation = intrinsicRate / np.std(VelSnipFast)
            independentModulation = intrinsicRate / np.std(independentSignal)
            noiseModulation = intrinsicRate / np.std(noiseSignal)

            ModExtent = 10
            velocityModulationAdjusted = (velocityModulation / maximumIteration) * ModExtent
            independentModulationAdjusted = (independentModulation / maximumIteration) * ModExtent / 2
            noiseModulationAdjusted = (noiseModulation / maximumIteration)

            r_t = intrinsicRate - (velocityModulationAdjusted * VelSnipFast) + (independentModulationAdjusted * independentSignal) + (noiseModulationAdjusted * noiseSignal)
            uncorrectedRate = np.mean(r_t)
            rateDifference = intrinsicRate - uncorrectedRate
            r_t += rateDifference

            # Downsample Independent Signal
            independentSignal_downsamp = independentSignal[::int(real_dt / sim_params.dt)]
            independentSignal_downsamp = independentSignal_downsamp[cutRise:]
            independentSignal_downsamp = independentSignal_downsamp[int(halfShiftRange):-int(halfShiftRange)]

            # Spike Generation
            timeWindow = np.arange(sim_params.t_start, sim_params.t_end, sim_params.dt)
            x_rand = np.random.rand(len(timeWindow))
            spikes = renewalFactor * r_t * sim_params.dt > x_rand

            spikeTimes = np.where(spikes == 1)[0]
            intervals = np.diff(spikeTimes)

            # Calcium Dynamics
            spikes_array = np.zeros_like(timeWindow)
            spikes_array[spikeTimes] = 1
            Ca_vec = np.zeros_like(spikes_array)
            Ca_vec[0] = max(sim_params.Ca_rest, spikes_array[0] * sim_params.spike_amp)

            for n in range(1, len(spikes_array)):
                dCa_dt_nom = (-sim_params.dt * sim_params.Ca_extrusion * (Ca_vec[n - 1] - sim_params.Ca_rest)) + (spikes_array[n] * sim_params.spike_amp)
                dCa_dt_denom = 1 + sim_params.Ca_binding_ratio + ((sim_params.indicator_conc * sim_params.Ca_affinity) / (Ca_vec[n - 1] + sim_params.Ca_affinity))**2
                dCa_dt = dCa_dt_nom / dCa_dt_denom
                Ca_vec[n] = Ca_vec[n - 1] + dCa_dt

            # Double Exponential Kernel
            TMP = np.convolve(Ca_vec - sim_params.Ca_rest, h, mode='full')[:len(Ca_vec)] + sim_params.Ca_rest
            Ca_bound = TMP[:len(Ca_vec)]

            # Hill Equation
            dF_F = sim_params.dynamic_range / (1 + (sim_params.K_D / Ca_bound)**sim_params.hill_coefficient)
            F = sim_params.F0 * dF_F + sim_params.F0

            SIM_record = {
                "DFF" : dF_F,
                "spikeTimes" : spikeTimes,
                "intervals" : intervals,
            }

            SIM.append(SIM_record)

            # SNR and Downsampled Signals
            signal_downsamp = dF_F[np.arange(0, dF_F.shape[0], int(real_dt/sim_params.dt))]
            noisySignal = signal_downsamp + (np.random.rand(len(signal_downsamp)) - 0.5)  # Add noise
            noisySignal = noisySignal[cutRise:]
            noisySignal = noisySignal[int(halfShiftRange):-int(halfShiftRange)]

            VelSnipDownsamp = VelSnipFast[::int(real_dt / sim_params.dt)]
            VelSnipDownsamp = VelSnipDownsamp[cutRise:]
            r_t_downsamp = r_t[::int(real_dt / sim_params.dt)][cutRise:]
            r_t_downsamp = r_t_downsamp[int(halfShiftRange):-int(halfShiftRange)]

            # Auto-correlation of Spikes
            spikeDownsamp = spikes[cutRise:][int(halfShiftRange):-int(halfShiftRange)]
            spikeACF = autocorr(spikeDownsamp, nlags=min(nlags - cutRise, 2000))

            # Auto-correlation of rate
            r_t_SD = np.std(r_t_downsamp)
            rateACF = autocorr(r_t_downsamp, nlags=min(nlags - cutRise, 2000))

            # Auto-correlation of noise
            noisySignal = zscore(noisySignal)
            acf = autocorr(noisySignal, nlags=min(nlags - cutRise, 2000))




            # Fit Exponential Decay to Spike ACF
            def exp2(x, a, b, c, d):
                return a * np.exp(b * x) + c * np.exp(d * x)

            lags = np.arange(len(spikeACF))
            spikeFit, _ = curve_fit(exp2, lags[:300], spikeACF[:300], p0=[1, -1, 0, 0])
            rtFit, _    = curve_fit(exp2, lags[:300], rateACF[:300], p0=[1, -1, 0, 0])
            exFit, _    = curve_fit(exp2, lags[:300], acf[:300], p0=[1, -1, 0, 0])

            SM = np.max(noisySignal)
            SV = np.var(noisySignal)

            record = {
                "trainN": trainN,
                "noiseModExtentIDX": noiseModExtentIDX,  # Replace with the loop index
                "spikeTau": (-1 / spikeFit[1]) * real_dt,  # `spikeFit[1]` corresponds to `b`
                "signal": noisySignal.tolist(),  # Convert signal to list if it's an array
                "acf": acf.tolist(),  # Convert ACF to list
                "NewTau": (-1 / exFit[1]) * (1 / 10),  # `exFit[1]` corresponds to `b`
                "r_t": r_t_downsamp.tolist(),
                "rateACF": rateACF.tolist(),
                "rateTau": (-1 / rtFit[1]) * (1 / 10),
                "velTerm": velocityModulationAdjusted,
                "rateSEM": r_t_SD,
                "VelSD": np.std(VelSnipFast)
            }

            data.append(record)


data_df = pd.DataFrame(data)
sim_df = pd.DataFrame(SIM)


plot_end = 100
plt.figure() #FOR RAW POISSON SPIKES
for trainN in range(10):
    plt.scatter(sim_df.spikeTimes[trainN][:plot_end], trainN*np.ones(len(sim_df.spikeTimes[trainN][:plot_end])), marker='|', color='black')
