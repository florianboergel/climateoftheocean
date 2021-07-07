import numpy as np
import xarray as xr

class ebm():
    """
    Zero order energy balance model
    """

    def __init__(self, T, t, deltat, CO2):
        self.T = np.array(T)
        self.t = t
        
        self.deltat = deltat
        self.C = 51.
        self.a = 5.
        self.B = -1.3
        self.co2_pi = 280.
        self.alpha = 0.3
        self.S = 1368.
        self.co2 = CO2
        self.CO2_PI = 280.
        self.A = 221.2

    def absorbed_solar_radiation(self):
        return (self.S*(1-self.alpha2)/4.)  # [W/m^2]
    def outgoing_thermal_radiation(self):
        if self.T.size == 1:
            return self.A - self.B*self.T
        else:
            return self.A - self.B*self.T[-1]
        
    def greenhouse_effect(self):
        if self.T.size == 1:
            return self.a*np.log(self.co2(self.t)/self.CO2_PI)
        else:
            return self.a*np.log(self.co2(self.t[-1])/self.CO2_PI)
        
    def tendency(self):
        if self.T.size == 1:
             return 1. / self.C * (
            + self.absorbed_solar_radiation()
            - self.outgoing_thermal_radiation()
            + self.greenhouse_effect()
            )
        else:
            return 1. / self.C * (
            + self.absorbed_solar_radiation()
            - self.outgoing_thermal_radiation()
            + self.greenhouse_effect()
            )        
        
    def run(self, end_year):
        for year in range(end_year):
            self.timestep()
     
    def timestep(self):
        if self.T.size == 1:
            self.alpha2 = calc_alpha(self.T, alpha0=self.alpha) # Added the function call here
            self.T = np.append(self.T, self.T + self.deltat * self.tendency())
            self.t = np.append(self.t, self.t + self.deltat)
        else:
            self.alpha2 = calc_alpha(self.T[-1], alpha0=self.alpha) # Added the function call here
            self.T = np.append(self.T, self.T[-1] + self.deltat * self.tendency())
            self.t = np.append(self.t, self.t[-1] + self.deltat)

def calc_alpha(T, alpha0, alphai = 0.5, deltaT=10.):
    if T < - deltaT:
        return alphai
    elif -deltaT <= T < deltaT:
        return alphai + (alpha0 - alphai)*(T + deltaT) / (2*deltaT)
    elif T >= deltaT:         
        return alpha0