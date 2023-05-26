"""Here will lie the wall temps. Early on we're just going to do throat temp just to get sizing down"""
import sys
sys.path.insert(1,"./")
import json
""" Cooldown analysis conops
Use thrust chamber to get flow props and store geometry
pass chamber props to setup function for whatever manufacturing technique
acquire cooling passage properties
send them to the trenches of steady state cooling
return heats
send them to the structural analysis
return margin of safety
iterate until we reach the ideal thickness
"""
import Toolbox.Constant as const
import scipy
import re as regex
import os
from rocketprops.rocket_prop import get_prop
import numpy as np
import math
from scipy import interpolate
import Toolbox.IsentropicEquations as Ise
import matplotlib.pyplot as plt
from Components.Component import Component
from scipy import optimize

class CoolingSystem(Component): #This object should be used to calculate stersses and factors of safety given geometry and temperatures

    def __init__(self,params,thrustchamber, xlist, rlist, chlist, chanelToLandRatio, twlist, nlist, helicitylist=None,
                                dxlist=None,  material = "inconel 718", setupMethod = None): #Initializes geometry and useful values for rectangular changels
        self.params = params
        self.thrustchamber = thrustchamber
        self.chanelToLandRatio = chanelToLandRatio
        #cooling system runs backwards through the chamber, so all the lists need to be flipped
        if xlist[0]-xlist[1] > 0:
            self.rlist = rlist
            self.xlist = xlist
            self.nlist = nlist
            self.chlist = chlist   
            self.twlist=twlist
            self.helicitylist = helicitylist
            self.dxlist = dxlist
        else:
            self.rlist = np.flip(rlist)
            self.xlist = np.flip(xlist)
            self.nlist = np.flip(nlist)
            self.chlist = np.flip(chlist)
            self.twlist = np.flip(twlist)
            self.helicitylist = np.flip(helicitylist)
            self.dxlist = np.flip(dxlist)
        self.material = material
        if setupMethod is None: #Use default, which is square helical channels.
            self.fixedRSquareChanelSetup()
        else:
            try:
                globals()[setupMethod]()
            except:
                print("Setup method passed does not exist")
        if self.helicitylist is None:
            self.helicitylist = np.ones(np.size(self.xlist)) * math.pi / 2

        if not (get_prop(self.params['fuelname']) is None):
            self.pObj = get_prop(self.params['fuelname'])
            self.ethpercent = None
            self.pObjWater = None
        else:
            try:
                print('Fuel does not exist, assuming its a water ethanol blend')
                self.ethpercent = int(regex.search(r'\d+', self.params['fuelname']).group()) / 100
                self.pObj = get_prop("ethanol")
                self.pObjWater = get_prop("water")
            except:
                print("rocketProps busted lol")

        try:
            self.matprops = json.load(self.material)
        except:
            print("either you didn't call this with a mat name or its not set up yet")
            self.kwInterpolator = interpolate.interp1d(
                np.hstack((-1000, 25, np.arange(100, 1001, 100) + 273.5, np.array([10000]))),
                np.array([0, 12.9, 13.9, 16.1, 18.2, 20, 22.1, 24.7, 31.4, 30.1, 31.5, 36.7,
                          36.7]), kind='linear')
            self.matprops = {'kw': self.kwInterpolator}  # copper = 398, mild steel = 51??
            self.matprops['rho'] = 8220  # kg/m3 for inconcel 718

            cpInterpolator = interpolate.interp1d(np.hstack((25, np.arange(100, 1001, 100) + 273.5, np.array([10000]))),
                                                  1000 * np.array(
                                                      [.537, .543, .577, .596, .608, .628, .665, .722, .756, .785, .874,
                                                       .874]), kind='linear')
            self.matprops['cp'] = cpInterpolator

        self.Trlist = np.zeros(self.xlist.size)
        for ind in np.arange(0, self.xlist.size):
            x = self.xlist[ind]
            self.Trlist[ind] = recoveryTemp(self.thrustchamber.tempInterpolator(self.xlist[ind]), self.params['gamma'],
                                            self.thrustchamber.machInterpolator(self.xlist[ind]),
                                            Pr=self.params['pr_throat'])

        self.staticnozzleparameters = {
            'throatdiameter': self.thrustchamber.rt * 2,
            'prandtlns': self.params['prns'],
            'viscosityns': self.params['viscosityns'],
            'cpns': self.params['cpns'],
            'pcns': self.params['pc'],
            'cstar': self.params['cstar'],
            'throatRadiusCurvature': self.params["throat_radius_curvature"],
            'at': self.thrustchamber.at,
            'tcns': self.params['temp_c'],
            'kw': self.matprops['kw'],
            'pObj': self.pObj,
            'pObjWater': self.pObjWater,
            'ethpercent': self.ethpercent,
            'roughness': 320 * 10 ** -6,
            'nlist': self.nlist,
            'helicitylist': self.helicitylist,
            'coatingthickness': .0005,
            'kcoating': 10.1,
            'hcoating': 14470
        }
        self.coolantpressurelist = np.zeros(self.xlist.size)
        self.Twglist = np.zeros(self.xlist.size)
        self.hglist = np.zeros(self.xlist.size)
        self.qdotlist = np.zeros(self.xlist.size)
        self.Qdotlist = np.zeros(self.xlist.size)
        self.fincoolingfactorlist = np.zeros(self.xlist.size)
        self.Twclist = np.zeros(self.xlist.size)
        self.hclist = np.zeros(self.xlist.size)
        self.Tclist = np.zeros(self.xlist.size)
        self.rholist = np.zeros(self.xlist.size)
        self.viscositylist = np.zeros(self.xlist.size)
        self.Relist = np.zeros(self.xlist.size)

        self.Vollist = 2 * math.pi * self.rlist / self.nlist * self.dxlist * self.twlist  # volume of the metal that is heating up per chanel

    def FOS(self, preslist): #Calculates stresses using just sigma_tangential from heister page 207
        self.rclist = np.zeros(
            self.rlist.size)  # Radius of curvature is defined such that a negative readius of curvature is convex, positive is concave (like the throat)
        ind = 1
        dr2dx2 = ( self.rlist[ind + 1] - 2 *  self.rlist[ind] +  self.rlist[ind - 1]) / ((( self.xlist[ind - 1] -  self.xlist[ind + 1]) / 2) ** 2)
        if dr2dx2 == 0:
            self.rclist[ind] = 0
        else:
            drdx = (( self.rlist[ind] -  self.rlist[ind - 1]) / ( self.xlist[ind] -self.xlist[ind - 1]) + (self.rlist[ind] - self.rlist[ind - +1]) / (
                       self.xlist[ind] -self.xlist[ind + 1])) / 2
            self.rclist[ind] = ((1 + drdx ** 2) ** (3 / 2)) / (dr2dx2)
        self.rclist[0] = self.rclist[1]
        while ind <self.xlist.size - 1:
            dr2dx2 = (self.rlist[ind + 1] - 2 * self.rlist[ind] + self.rlist[ind - 1]) / (((self.xlist[ind - 1] -self.xlist[ind + 1]) / 2) ** 2)
            if dr2dx2 == 0:
                self.rclist[ind] = 0
            else:
                drdx = ((self.rlist[ind] - self.rlist[ind - 1]) / (self.xlist[ind] -self.xlist[ind - 1]) + (
                            self.rlist[ind] - self.rlist[ind - +1]) / (self.xlist[ind] -self.xlist[ind + 1])) / 2
                self.rclist[ind] = ((1 + drdx ** 2) ** (3 / 2)) / (dr2dx2)
            ind = ind + 1
        self.rclist[-1] = self.rclist[-2]

        # CURRENTLY HAVE THIS AS FIXED, NEED TO BE ABLE TO CHANGE THIS WITH PRELOADED MATPROPS
        self.EInterpolator = interpolate.interp1d(
            ((np.hstack((70, np.arange(100, 2001, 100), np.array([10000]))) - 32) * 5 / 9) + 273.15,
            np.array(
                [29, 28.8, 28.4, 28, 27.6, 27.1, 26.7, 26.3, 25.8, 25.3, 24.8, 24.2, 23.7, 23.0, 22.3, 21.3, 20.2, 18.8,
                 17.4, 15.9, 14.3, 0]) * (10 ** 6) * const.psiToPa,
            kind='linear')  # https://www.engineersedge.com/materials/inconel_718_modulus_of_elasticity_table_13562.htm
        self.PoissonInterpolator = interpolate.interp1d(
            ((np.hstack((70, np.arange(100, 2001, 100), np.array([10000]))) - 32) * 5 / 9) + 273.15,
            np.array(
                [.294, .291, .288, .280, .275, .272, .273, .271, .272, .271, .276, .283, .292, .306, .321, .331, .334,
                 .341, .366, .402, .402, .402]),
            kind='linear')  # https://www.engineersedge.com/materials/inconel_718_modulus_of_elasticity_table_13562.htm
        # self.SyInterpolator = interpolate.interp1d(np.array([25,425,650,870,871,950,10000]),
        #                                      np.array([1200,1050,1000,300,290,100,1])*10**6, kind='linear') # pg 734 metal additive manufacturing, inconel 718
        self.SyInterpolator = interpolate.interp1d(np.array([25, 425, 650, 870, 871, 10000]) + 273,
                                                   np.array([1200, 1050, 1000, 300, 10, 1]) * 10 ** 6,
                                                   kind='linear')  # pg 734 metal additive manufacturing, inconel 718

        self.AlphaInterpolator = interpolate.interp1d(np.hstack((np.arange(0, 1001, 100), np.array([10000]))) + 273,
                                                      np.array(
                                                          [12.6, 12.6, 13.9, 14.2, 14.5, 14.8, 15.1, 15.6, 16.4, 17.5,
                                                           17.8, 17.8]) * 10 ** (-6),
                                                      kind='linear')  # pg 781 metal additive manufacturing, inconel 718

        dplist = self.coolantpressurelist-preslist
        avgtemplist = (self.Twglist+self.Twclist)/2
        deltaTlist = self.Twglist-self.Twclist
        Elist = self.EInterpolator(avgtemplist)
        poissonlist = self.PoissonInterpolator(avgtemplist)
        Sylist = self.SyInterpolator(avgtemplist)
        Alphalist = self.AlphaInterpolator(avgtemplist)
        FUDGEFACTORFORTHERMALSTRESS = 1#.02 #pretty much disregarding it rn lol
        self.sigmatangentiallist = dplist/2*((self.cwlist/self.twlist)**2)   +  FUDGEFACTORFORTHERMALSTRESS*Elist*Alphalist*deltaTlist/(2*(1-poissonlist))
        self.FOSlist = Sylist/self.sigmatangentiallist

    def fixedRSquareChanelSetup(self):  # MAKE SURE TO PASS SHIT  ALREADY FLIPPED
        if self.helicitylist is None:
            self.helicitylist = np.ones(np.size(self.xlist)) * math.pi / 2
        if self.dxlist is None:
            self.dxlist = np.ones(np.size(self.xlist))
            index = 1
            while index < np.size(self.xlist):
                self.dxlist[index] = abs(self.xlist[index - 1] - self.xlist[index])
                index = index + 1
            self.dxlist[0] = self.dxlist[
                1]  # this is shit, but I actuall calculate an extra spatial step at the start for some reason, so our CC is 1 dx too long. Makes the graphs easier to work with tho lol, off by one error be damned

        # FOR NOW THIS IS ASSUMING SQUARE CHANELS!
        # \     \  pavail\     \ 
        # \     \ |----| \     \
        #  \     \        \     \
        #   \     \        \     \
        # sin(helicitylist) pretty much makes sure that we are using the paralax angle to define the landwidth
        self.perimavailable = math.pi * 2 * (self.rlist + self.twlist) / self.nlist * np.sin(self.helicitylist)
        self.landwidthlist = self.perimavailable / (self.chanelToLandRatio + 1)
        self.cwlist = self.perimavailable - self.landwidthlist

        self.alist = self.chlist * self.cwlist
        self.salist = self.cwlist / self.dxlist
        self.vlist = self.params['mdot_fuel'] / self.params['rho_fuel'] / self.alist / self.nlist
        # if np.min(cwlist/chlist)>10:
        #    raise Exception(f"Aspect Ratio is crazyyyyy")

        self.hydraulicdiamlist = 4 * self.alist / (2 * self.chlist + 2 * self.cwlist)
        self.coolingfactorlist = np.ones(self.xlist.size)
        self.heatingfactorlist = np.ones(self.xlist.size) * .6  # .6 is from cfd last year, i think its bs but whatever
        """Fin cooling factor func is 2*nf*CH+CW. nf is calculated as tanh(mL)/ml. Ml is calculated as sqrt(hpL^2/ka),
        h=hc, P=perimeter in contact with coolant = dx, A = dx*landwidth/2 (assuming only half the fin works on the coolant, 2* factor in other spot,
        I think I can do that because of axisymmetric type), k=kw, L=height of fin = chanel height
        This is all from "A Heat Transfer Textbook, around page 166 and onwards."""
        self.fincoolingfactorfunc = lambda hc, kw, ind: (math.tanh(
            self.chlist[ind] * math.sqrt(2 * hc / kw * self.landwidthlist[ind])) / \
                                                    (self.chlist[ind] * math.sqrt(2 * hc / kw * self.landwidthlist[ind]))) * 2 * \
                                                   self.chlist[ind] + self.cwlist[ind]


        
    

        """Twglist, hglist, Qdotlist, Twclist, hclist, Tclist, coolantpressurelist, qdotlist, fincoolingfactorlist, rholist, viscositylist, Relist = CS.steadyStateTemperatures(
            None, TC, params, salistflipped, nlist, coolingfactorlist,
            heatingfactorlist,self.xlist, vlistflipped, 293, params['pc'] + params['pc'] * .2 + 50 * const.psiToPa,
            twlistflipped, hydraulicdiamlist, rgaslist=rlist, fincoolingfactorfunc=fincoolingfactorfunc, dxlist=dxlist)

        material = "inconel 715"
        Structure = CS.StructuralAnalysis(self.rlist,self.xlist, nlist, chlist, cwlist, twlist, material)
        FOSlist = Structure.FOS(Twglist, Twclist, coolantpressurelist, preslist)
       self.xlist = np.flip(
           self.xlist)  # idk whats flipping this haha but its something in the steadystatetemps function, so we have to flip it back
        return alistflipped,self.xlist, vlistflipped, \
               hydraulicdiamlist, salistflipped, dxlist, fincoolingfactorfunc, cwlist, \
               Twglist, hglist, Qdotlist, Twclist, hclist, Tclist, coolantpressurelist, qdotlist, fincoolingfactorlist, rholist, viscositylist, Relist, \
               FOSlist"""

    
    


    def Equilibirum(self, initialcoolanttemp, initialcoolantpressure):
        
    

        Tc = initialcoolanttemp
        coolantpressure = initialcoolantpressure

        
        
        
        for ind in np.arange(0,self.xlist.size):
            self.Tclist[ind] = Tc  # set cooland at current station to coolant temp
            dx = 0
            if ind == 0:
                dx =self.xlist[ind] -self.xlist[ind + 1]
                coolantpressure = coolantpressure - 4 * cf(Tc, coolantpressure, self.vlist[ind], self.hydraulicdiamlist[ind],
                                                        self.staticnozzleparameters, roughness=self.staticnozzleparameters['roughness']) * (
                                        dx) / \
                                self.hydraulicdiamlist[ind] * (
                                            .5 * rho(Tc, coolantpressure, self.staticnozzleparameters) * self.vlist[ind] ** 2)
            else:
                dx =self.xlist[ind - 1] -self.xlist[ind]
                coolantpressure = coolantpressure - 4 * cf(Tc, coolantpressure, self.vlist[ind], self.hydraulicdiamlist[ind],
                                                        self.staticnozzleparameters, roughness=self.staticnozzleparameters['roughness']) * (
                                        dx) / \
                                self.hydraulicdiamlist[ind] * (
                                        .5 * rho(Tc, coolantpressure, self.staticnozzleparameters) * self.vlist[ind] ** 2)


            self.coolantpressurelist[ind] = coolantpressure
            x =self.xlist[ind]
            if ind%10 == 0:
                print(x)
            Tri = self.Trlist[ind]
            gassidearea = self.thrustchamber.areaInterpolator(x)
            rgas = self.rlist[ind]
            fincoolingfactorfunc_atstation = lambda hc, kw : self.fincoolingfactorfunc(hc,kw,ind)

            self.Twglist[ind] = QdotdiffMinimizer(self.staticnozzleparameters, gassidearea, self.thrustchamber.machInterpolator(x),
                                            self.params['gamma'],Tri, Tc, self.twlist[ind], coolantpressure, self.vlist[ind], self.hydraulicdiamlist[ind],
                                            self.coolingfactorlist[ind],self.heatingfactorlist[ind], rgas, self.nlist[ind], fincoolingfactorfunc_atstation,
                                            helicity = self.staticnozzleparameters['helicitylist'][ind])
            self.hglist[ind] = self.heatingfactorlist[ind]*Bartz(self.staticnozzleparameters['throatdiameter'], self.staticnozzleparameters['viscosityns'],
                                self.staticnozzleparameters['prandtlns'], self.staticnozzleparameters['cpns'],
                                self.staticnozzleparameters['pcns'], self.staticnozzleparameters['cstar'],
                                self.staticnozzleparameters['throatRadiusCurvature'], self.staticnozzleparameters['at'],
                                gassidearea, self.Twglist[ind], self.staticnozzleparameters['tcns'],
                                self.thrustchamber.machInterpolator(x), self.params['gamma'])
            self.Qdotlist[ind] = self.hglist[ind] * (self.Trlist[ind] - self.Twglist[ind])*2*math.pi*rgas/self.nlist[ind]
            self.qdotlist[ind] = self.hglist[ind] * (self.Trlist[ind] - self.Twglist[ind])
            # This works since its the same method used in qdotdiff function, its just a one iteration approx
            self.Twclist[ind] = self.Twglist[ind] - self.Qdotlist[ind]/(2*math.pi*rgas/self.nlist[ind]) / (self.matprops['kw']((self.Twglist[ind])) / self.twlist[ind])
            self.Twclist[ind] = self.Twglist[ind] - self.Qdotlist[ind]/(2*math.pi*rgas/self.nlist[ind]) / (self.matprops['kw']((self.Twglist[ind]+self.Twclist[ind])/2) / self.twlist[ind])
            self.hclist[ind] = hc(self.Twclist[ind], Tc, coolantpressure, self.vlist[ind], self.hydraulicdiamlist[ind],
                            self.staticnozzleparameters['pObj'], self.staticnozzleparameters['pObjWater'],
                            self.staticnozzleparameters['ethpercent'], self.coolingfactorlist[ind])
            self.fincoolingfactorlist[ind] = dx*fincoolingfactorfunc_atstation(self.hclist[ind],(self.matprops['kw']((self.Twglist[ind]+self.Twclist[ind])/2)))\
                                        /math.sin(self.staticnozzleparameters['helicitylist'][ind])/(gassidearea/self.nlist[ind])
            #This is the ratio of corrected coolant side area over gas side area servicied by coolant passage
            #This would be (2*ch*finefficiency + cw)*(dx/sin(helicity)) / (gassidearea/numchanels)
            Tc += self.Qdotlist[ind] * self.dxlist[ind] / (self.params['mdot_fuel']/self.nlist[ind] * heatCapacity(Tc, self.staticnozzleparameters))
            self.rholist[ind] = rho(Tc, coolantpressure, self.staticnozzleparameters)
            self.viscositylist[ind]= viscosity(Tc,coolantpressure,  self.staticnozzleparameters['pObj'],self.staticnozzleparameters['pObjWater'],self.staticnozzleparameters['ethpercent'])
            self.Relist[ind] = reynolds(Tc,coolantpressure,self.vlist[ind],self.hydraulicdiamlist[ind], self.staticnozzleparameters['pObj'],
                                        self.staticnozzleparameters['pObjWater'], self.staticnozzleparameters['ethpercent'])
            

    def ablative(self):
        return "set this up"
    
    
    def Step(self,dt,initialcoolanttemp, initialcoolantpressure):

        Tc = initialcoolanttemp
        coolantpressure = initialcoolantpressure


        for ind in np.arange(0,self.xlist.size):
            self.Tclist[ind] = Tc  # set cooland at current station to coolant temp
            dx = 0
            if ind == 0:
                dx = self.xlist[ind] - self.xlist[ind + 1]
                coolantpressure = coolantpressure - 4 * cf(Tc, coolantpressure, self.vlist[ind],
                                                           self.hydraulicdiamlist[ind],
                                                           self.staticnozzleparameters,
                                                           roughness=self.staticnozzleparameters['roughness']) * (
                                      dx) / \
                                  self.hydraulicdiamlist[ind] * (
                                          .5 * rho(Tc, coolantpressure, self.staticnozzleparameters) * self.vlist[
                                      ind] ** 2)
            else:
                dx = self.xlist[ind - 1] - self.xlist[ind]
                coolantpressure = coolantpressure - 4 * cf(Tc, coolantpressure, self.vlist[ind],
                                                           self.hydraulicdiamlist[ind],
                                                           self.staticnozzleparameters,
                                                           roughness=self.staticnozzleparameters['roughness']) * (
                                      dx) / \
                                  self.hydraulicdiamlist[ind] * (
                                          .5 * rho(Tc, coolantpressure, self.staticnozzleparameters) * self.vlist[
                                      ind] ** 2)

            self.coolantpressurelist[ind] = coolantpressure
            x = self.xlist[ind]
            if ind % 10 == 0:
                print(x)
            Tri = self.Trlist[ind]
            gassidearea = self.thrustchamber.areaInterpolator(x)
            rgas = self.rlist[ind]
            fincoolingfactorfunc_atstation = lambda hc, kw : self.fincoolingfactorfunc(hc,kw,ind)

            self.hglist[ind] = self.heatingfactorlist[ind]*Bartz(self.staticnozzleparameters['throatdiameter'], self.staticnozzleparameters['viscosityns'],
                                self.staticnozzleparameters['prandtlns'], self.staticnozzleparameters['cpns'],
                                self.staticnozzleparameters['pcns'], self.staticnozzleparameters['cstar'],
                                self.staticnozzleparameters['throatRadiusCurvature'], self.staticnozzleparameters['at'],
                                gassidearea, self.Twglist[ind], self.staticnozzleparameters['tcns'],
                                self.thrustchamber.machInterpolator(x), self.params['gamma'])

            self.Qdotlist[ind] = self.hglist[ind] * (Tri - self.Twglist[ind])*2*math.pi*rgas/self.nlist[ind] #This is Qdot per unit length, so it should be multiplied by dx to get actual total heat flux. This overcomplicates solver
            Qdot_throughwall = abs(self.staticnozzleparameters['kw']((self.Twglist[ind]+self.Twclist[ind])/2)/self.twlist[ind]*(self.Twglist[ind]-self.Twclist[ind]))*2*math.pi*rgas/self.nlist[ind]

            #update twglist by mupltipliyng Qdot by dt, then divide by heat capacity and mass CURRENTLY ASSUMING MASS IS HALF OF TOTAL FOR CHANEL
            self.Twglist[ind] =self.Twglist[ind] + dt*(self.Qdotlist[ind] - Qdot_throughwall)*self.dxlist[ind] /(self.Vollist[ind]/2*self.matprops['rho']* self.matprops['cp']((self.Twglist[ind]+self.Twclist[ind])/2))

            #Check huzel and huang page 98 for graph of effect of helicity on hc
            turningangle = math.pi/2-self.helicitylist[ind]
            curvature_enhancement_factor =  np.max([1,np.min([1.4/18*turningangle*180/math.pi-1.4/18*20+1, 1.4,
            -1.4/20*turningangle*180/math.pi+1.4/20*80])])#cheesy linear interp
            hci = curvature_enhancement_factor*hc(self.Twclist[ind], Tc, coolantpressure, self.vlist[ind], self.hydraulicdiamlist[ind], self.staticnozzleparameters['pObj'],
                    self.staticnozzleparameters['pObjWater'], self.staticnozzleparameters['ethpercent'], self.coolingfactorlist[ind]) #only reason we guess twci now is to get hci, otherwise we are calculating the thermal resistance directly from twg to tc
            nf = fincoolingfactorfunc_atstation(hci,self.staticnozzleparameters['kw']((self.Twglist[ind]+self.Twclist[ind])/2)) # THIS IS FIN EFFICIENCY * 2 * CH + CW , if you dont want to factor in fin cooling just set this equal to cw
            Rtotal = (self.twlist[ind] / self.staticnozzleparameters['kw']((self.Twglist[ind]+self.Twclist[ind])/2)       )*2*math.pi*rgas/self.nlist[ind]+1/(nf*hci)*math.sin(self.helicitylist[ind]) #missing a factor of 1/dx, to be consistent with Qdotguess being per unit length
            # dividing by the sin of helicity here implies that the fin is acting over dx/sin(helicity), which it is!
            Qdothc = 1/Rtotal * (self.Twclist[ind]-Tc) #/ math.sin(helicity) #smaller helicity means longer chanel length per dx, resulting in more heat flux into coolant

            self.Twclist[ind] =self.Twclist[ind] + dt*(Qdot_throughwall - Qdothc)*self.dxlist[ind] /(self.Vollist[ind]/2*self.matprops['rho']* self.matprops['cp']((self.Twglist[ind]+self.Twclist[ind])/2))

            self.qdotlist[ind] = self.hglist[ind] * (self.Trlist[ind] - self.Twglist[ind])
            # This works since its the same method used in qdotdiff function, its just a one iteration approx

            self.hclist[ind] = hci
            self.fincoolingfactorlist[ind] = dx*fincoolingfactorfunc_atstation(self.hclist[ind],(self.matprops['kw']((self.Twglist[ind]+self.Twclist[ind])/2)))\
                                        /math.sin(self.staticnozzleparameters['helicitylist'][ind])/(gassidearea/self.nlist[ind])
            #This is the ratio of corrected coolant side area over gas side area servicied by coolant passage
            #This would be (2*ch*finefficiency + cw)*(dx/sin(helicity)) / (gassidearea/numchanels)
            Tc +=  Qdothc * self.dxlist[ind] / (self.params['mdot_fuel']/self.nlist[ind] * heatCapacity(Tc, self.staticnozzleparameters))
            self.rholist[ind] = rho(Tc, coolantpressure, self.staticnozzleparameters)
            self.viscositylist[ind]= viscosity(Tc,coolantpressure,  self.staticnozzleparameters['pObj'],self.staticnozzleparameters['pObjWater'],self.staticnozzleparameters['ethpercent'])
            self.Relist[ind] = reynolds(Tc,coolantpressure,self.vlist[ind],self.hydraulicdiamlist[ind], self.staticnozzleparameters['pObj'],
                                        self.staticnozzleparameters['pObjWater'], self.staticnozzleparameters['ethpercent'])

    def GenerateCad(self, outputFilePath, externalWall, method=None): #externalWall is how far away the wall is from the top of the chanels, can be a number or a list with size xlist.size
        if method is None:
            xlistnew, ylistnew, zlistnew, hydraulicdiamlistnew, chlistnew = ChanelBean(self.xlist, self.rlist,
                                                                                       self.twlist,
                                                                                       self.helicitylist, self.chlist,
                                                                                       self.cwlist)
        else:
            try: #feel fre to add methods as functions below, chanle box corners already exists and is useful for CFD as it just creates the guiding curves for the corners of a box
                xlistnew, ylistnew, zlistnew, hydraulicdiamlistnew, chlistnew = globals()[method]()
            except:
                print("CAD method passed does not exist")

        if isinstance(externalWall,float) or isinstance(externalWall,int):
            ewlist = np.ones(self.xlist.size)*externalWall
        else:
            try:
                if externalWall.size == self.xlist.size:
                    ewlist = externalWall
                else:
                    print(f"Length of list incorrect. You passed an external wall list of len {externalWall.size}, needs to be {self.xlist.size}")
            except:
                print("invalid argument passed for externalWall")

        for index in range(0, xlistnew.shape[0]):
            with open(os.path.join(outputFilePath, f"ThrustChamber_Curve{index}.sldcrv"), "w") as f:
                for i in range(len(xlistnew[1, :])):
                    # print(f"{xchanel[i]} {ychanel[i]} {zchanel[i]}", file=f) # this if for olivers axis's
                    print(f"{ylistnew[index, i]} {xlistnew[index, i]} {zlistnew[index, i]}",
                          file=f)  # this if for roberts axis's

        with open(os.path.join(outputFilePath, "internalradius.sldcrv"), "w") as f:
            for i in range(len(self.xlist)):
                print(f"{self.xlist[i]} {self.rlist[i]} {0}", file=f)
        newxlist, externalrlist = rlistExtender(self.xlist, self.rlist, ewlist + np.flip(self.chlist) + np.flip(self.twlist))
        with open(os.path.join(outputFilePath, "externalradius.sldcrv"), "w") as f:
            for i in range(len(self.xlist)):
                print(f"{newxlist[i]} {externalrlist[i]} {0}", file=f)
# ns stands for nozzle start, use finite area combustor CEA to find it (or just use pc and tcomb)
# This equation neglects g because you dont need it for si units (i think)
def Bartz(throatdiameter, viscosityns, prandtlns, cpns, pcns, cstar, throatRadiusCurvature, at, a, twg, tcns, mach,
          gamma):
    sigma = ((.5 * twg / tcns * (1 + (mach ** 2) * (gamma - 1) / 2) + .5) ** (-.68)) * (
            1 + (mach ** 2) * (gamma - 1) / 2) ** (-.12)
    return (.026 / (throatdiameter ** .2) * (cpns * viscosityns ** .2) / (prandtlns ** .6) * (pcns / cstar) ** (.8) * (
            throatdiameter / throatRadiusCurvature) ** .1) * (at / a) ** .9 * sigma


def recoveryTemp(temp, gam, mach, Pr=None):
    Taw = Ise.totalT(temp, gam, mach) / (1 + (((gam - 1) / 2) * (mach ** 2)))
    if Pr is None:
        Pr = .6865
    r = Pr ** (1 / 3)
    return Taw * (1 + (r * (mach ** 2) * ((gam - 1) / 2)))


def qdotdiffMinimizer(staticnozzleparams, a, mach, gamma,
                      Tri, Tc, tw, coolantpressure, coolantvelocity, hydraulicdiam, coolingfactor, heatingfactor):
    return scipy.optimize.minimize_scalar(qdotdiff, bounds=(350, Tri), tol=10, method='bounded',
                                          args=(staticnozzleparams, a, mach, gamma,
                                                Tri, Tc, tw, coolantpressure, coolantvelocity, hydraulicdiam,
                                                coolingfactor, heatingfactor))[
        'x']  # going to find root of qdotdiff for twg


def QdotdiffMinimizer(staticnozzleparams, a, mach, gamma,
                      Tri, Tc, tw, coolantpressure, coolantvelocity, hydraulicdiam, coolingfactor, heatingfactor, rgas,
                      n, fincoolingfactorfunc,
                      helicity=math.pi / 2):
    return scipy.optimize.minimize_scalar(Qdotdiff, bounds=(350, Tri), tol=2, method='bounded',
                                          args=(staticnozzleparams, a, mach, gamma,
                                                Tri, Tc, tw, coolantpressure, coolantvelocity, hydraulicdiam,
                                                coolingfactor, heatingfactor, rgas, n, fincoolingfactorfunc, helicity))[
        'x']  # going to find root of qdotdiff for twg


def qdotdiff(Twgi, staticnozzleparams, a, mach, gamma,
             Tri, Tc, tw, coolantpressure, coolantvelocity, hydraulicdiam, coolingfactor, heatingfactor):
    hgi = heatingfactor * Bartz(staticnozzleparams['throatdiameter'], staticnozzleparams['viscosityns'],
                                staticnozzleparams['prandtlns'], staticnozzleparams['cpns'],
                                staticnozzleparams['pcns'], staticnozzleparams['cstar'],
                                staticnozzleparams['throatRadiusCurvature'],
                                staticnozzleparams['at'],
                                a, Twgi, staticnozzleparams['tcns'], mach, gamma)
    qdotguess = hgi * (Tri - Twgi)
    Twci = Twgi - ((qdotguess * tw) / staticnozzleparams['kw'](Twgi))  # get the initial guess
    Twci = Twgi - ((qdotguess * tw) / staticnozzleparams['kw']((Twgi + Twci) / 2))  # now use avg wall temp
    hci = hc(Twci, Tc, coolantpressure, coolantvelocity, hydraulicdiam, staticnozzleparams['pObj'],
             staticnozzleparams['pObjWater'], staticnozzleparams['ethpercent'], coolingfactor)
    qdothc = hci * (Twci - Tc)
    return abs(qdothc - qdotguess)


def Qdotdiff(Twgi, staticnozzleparams, a, mach, gamma,
             Tri, Tc, tw, coolantpressure, coolantvelocity, hydraulicdiam, coolingfactor, heatingfactor, rgas, n,
             fincoolingfactorfunc,
             helicity=math.pi / 2):
    hgi = heatingfactor*Bartz(staticnozzleparams['throatdiameter'], staticnozzleparams['viscosityns'],
                staticnozzleparams['prandtlns'], staticnozzleparams['cpns'],
                staticnozzleparams['pcns'], staticnozzleparams['cstar'], staticnozzleparams['throatRadiusCurvature'],
                staticnozzleparams['at'],
                a, Twgi, staticnozzleparams['tcns'], mach, gamma)
    hgi = 1/( 1/hgi) #lol
    Qdotguess = hgi * (Tri - Twgi)*2*math.pi*rgas/n #This is Qdot per unit length, so it should be multiplied by dx to get actual total heat flux. This overcomplicates solver
    Twci = Twgi - ((Qdotguess/(2*math.pi*rgas/n) * tw) / staticnozzleparams['kw'](Twgi))# get the initial guess
    kwi = staticnozzleparams['kw']((Twgi+Twci)/2)
    Twci = Twgi - ((Qdotguess/(2*math.pi*rgas/n) * tw) / staticnozzleparams['kw']((Twgi+Twci)/2)) # now use avg wall temp
    
    #Check huzel and huang page 98 for graph of effect of helicity on hc
    turningangle = math.pi/2-helicity
    curvature_enhancement_factor =  np.max([1,np.min([1.4/18*turningangle*180/math.pi-1.4/18*20+1, 1.4,
     -1.4/20*turningangle*180/math.pi+1.4/20*80])])#cheesy linear interp
    hci = curvature_enhancement_factor*hc(Twci, Tc, coolantpressure, coolantvelocity, hydraulicdiam, staticnozzleparams['pObj'],
             staticnozzleparams['pObjWater'], staticnozzleparams['ethpercent'], coolingfactor) #only reason we guess twci now is to get hci, otherwise we are calculating the thermal resistance directly from twg to tc
    nf = fincoolingfactorfunc(hci,kwi) # THIS IS FIN EFFICIENCY * 2 * CH + CW , if you dont want to factor in fin cooling just set this equal to cw
    Rtotal = (tw / (kwi)       )*2*math.pi*rgas/n+1/(nf*hci)/math.sin(helicity)
            #missing a factor of 1/dx, to be consistent with Qdotguess being per unit length
            # NEED TO FIGUREO UT HOW TO FACTOR IN HELICITY! Shoudl probably divide by sin.
            # dividing by the sin of helicity here implies that the fin is acting over dx/sin(helicity), which it is!
    Qdothc = 1/Rtotal * (Twci-Tc) #/ math.sin(helicity) #smaller helicity means longer chanel length per dx, resulting in more heat flux into coolant
    #print(f"{[Twgi, hgi, hci, nf, Qdotguess, Qdothc, abs(Qdothc - Qdotguess)]},")
    return abs(Qdothc - Qdotguess)


def hc(tempwall, temp, pres, fluidvelocity, hydraulicdiam, pObj, pObjWater, ethpercent, coolingfactor):
    Re = reynolds(temp, pres, fluidvelocity, hydraulicdiam, pObj, pObjWater, ethpercent)
    Pr = prandtl(temp, pres, pObj, pObjWater, ethpercent)
    if ethpercent is None:
        thermalcond = pObj.CondAtTdegR(temp * const.degKtoR) * const.BTUperHrFtRtoWperMK
        viscosity = .1 * pObj.Visc_compressed(temp * const.degKtoR,
                                              pres / const.psiToPa)
        viscosityw = .1 * pObj.Visc_compressed(tempwall * const.degKtoR,
                                               pres / const.psiToPa)
        if math.isnan(viscosityw):
            viscosityw = viscosity  # make this a non factor since it keeps breaking
    else:
        thermalcondeth = pObj.CondAtTdegR(temp * const.degKtoR) * const.BTUperHrFtRtoWperMK
        thermalcondwater = pObjWater.CondAtTdegR(temp * const.degKtoR) * const.BTUperHrFtRtoWperMK
        thermalcond = (ethpercent * thermalcondeth + (1 - ethpercent) * thermalcondwater)
        viscosityeth = .1 * pObj.Visc_compressed(temp * const.degKtoR,
                                                 pres / const.psiToPa)
        viscositywater = .1 * pObjWater.Visc_compressed(temp * const.degKtoR,
                                                        pres / const.psiToPa)
        viscosityethw = .1 * pObj.Visc_compressed(tempwall * const.degKtoR,
                                                  pres / const.psiToPa)
        viscositywaterw = .1 * pObjWater.Visc_compressed(tempwall * const.degKtoR,
                                                         pres / const.psiToPa)
        viscosity = ethpercent * viscosityeth + (1 - ethpercent) * viscositywater
        viscosityw = ethpercent * viscosityethw + (1 - ethpercent) * viscositywaterw
        if math.isnan(viscosityw):
            viscosityw = viscosity  # make this a non factor since it keeps breaking
    # FIGURE OUT THESE COEFFICIENTS PETE, PAGE 197 IN HEISTER eider - tate equation
    a = .027  # this is the weirtd one, its just a multiplier. Maybe include thisi n the cooling factor list? it depends on coolabnt, heister page 198
    m = .8
    n = .5  # .4
    b = .114
    # if pObj.PvapAtTdegR(temp * const.degKtoR) * const.psiToPa > pres:  # THIS MEANS YOUR ETHNAOL IS BOILING
    #    coolingfactor = coolingfactor / 10  # this is made up but hopefully will make it obvious your shit is boiling
    return coolingfactor * thermalcond / hydraulicdiam * a * Re ** m * Pr ** n * (viscosity / viscosityw) ** b


def reynolds(temp, pres, fluidvelocity, hydraulicdiameter, pObj, pObjWater, ethpercent):
    if ethpercent is None:
        rho = 1000 * pObj.SG_compressed(temp * const.degKtoR,
                                        pres / const.psiToPa)
        viscosity = .1 * pObj.Visc_compressed(temp * const.degKtoR,
                                              pres / const.psiToPa)
    else:
        rhoeth = 1000 * pObj.SG_compressed(temp * const.degKtoR,
                                           pres / const.psiToPa)
        viscosityeth = .1 * pObj.Visc_compressed(temp * const.degKtoR,
                                                 pres / const.psiToPa)
        rhowater = 1000 * pObjWater.SG_compressed(temp * const.degKtoR,
                                                  pres / const.psiToPa)
        viscositywater = .1 * pObjWater.Visc_compressed(temp * const.degKtoR,
                                                        pres / const.psiToPa)
        viscosity = ethpercent * viscosityeth + (1 - ethpercent) * viscositywater
        rho = ethpercent * rhoeth + (1 - ethpercent) * rhowater
    return rho * fluidvelocity * hydraulicdiameter / viscosity


# W/mK, kg/m3, J/kgK
def prandtl(temp, pres, pObj, pObjWater, ethpercent):
    if ethpercent is None:
        rho = 1000 * pObj.SG_compressed(temp * const.degKtoR,
                                        pres / const.psiToPa)
        thermalcond = pObj.CondAtTdegR(temp * const.degKtoR) * const.BTUperHrFtRtoWperMK
        cp = pObj.CpAtTdegR(temp * const.degKtoR) * const.BTUperLbmRtoJperKgK
        thermaldiffusivity = thermalcond / (rho * cp)  # this the definition
        viscosity = .1 * pObj.Visc_compressed(temp * const.degKtoR,
                                              pres / const.psiToPa)  # the .1 is because its giving it in units of poise not SI
    else:
        rhoeth = 1000 * pObj.SG_compressed(temp * const.degKtoR,
                                           pres / const.psiToPa)
        viscosityeth = .1 * pObj.Visc_compressed(temp * const.degKtoR,
                                                 pres / const.psiToPa)
        thermalcondeth = pObj.CondAtTdegR(temp * const.degKtoR) * const.BTUperHrFtRtoWperMK
        cpeth = pObj.CpAtTdegR(temp * const.degKtoR) * const.BTUperLbmRtoJperKgK
        rhowater = 1000 * pObjWater.SG_compressed(temp * const.degKtoR,
                                                  pres / const.psiToPa)
        viscositywater = .1 * pObjWater.Visc_compressed(temp * const.degKtoR,
                                                        pres / const.psiToPa)
        thermalcondwater = pObjWater.CondAtTdegR(temp * const.degKtoR) * const.BTUperHrFtRtoWperMK
        cpwater = pObjWater.CpAtTdegR(temp * const.degKtoR) * const.BTUperLbmRtoJperKgK
        thermaldiffusivity = (ethpercent * thermalcondeth + (1 - ethpercent) * thermalcondwater) / (
                (ethpercent * rhoeth + (1 - ethpercent) * rhowater) * (
                ethpercent * cpeth + (1 - ethpercent) * cpwater))  # this the definition
        viscosity = ethpercent * viscosityeth + (1 - ethpercent) * viscositywater
        rho = ethpercent * rhoeth + (1 - ethpercent) * rhowater
    return viscosity / thermaldiffusivity / rho  # viscosity over rho to go from dyunamic vsicosity to kinematic viscosity (momentum diffusivity)
    # return viscosity*((ethpercent * cpeth + (1 - ethpercent) * cpwater))/ (ethpercent * thermalcondeth + (1 - ethpercent) * thermalcondwater)


def heatCapacity(temp, staticnozzleparameters):
    if staticnozzleparameters['ethpercent'] is None:
        return staticnozzleparameters['pObj'].CpAtTdegR(temp * const.degKtoR) * const.BTUperLbmRtoJperKgK
    else:
        cpeth = staticnozzleparameters['pObj'].CpAtTdegR(temp * const.degKtoR) * const.BTUperLbmRtoJperKgK
        cpwater = staticnozzleparameters['pObjWater'].CpAtTdegR(temp * const.degKtoR) * const.BTUperLbmRtoJperKgK
        return (staticnozzleparameters['ethpercent'] * cpeth + (1 - staticnozzleparameters['ethpercent']) * cpwater)


def rho(temp, pres, staticnozzleparameters):
    if staticnozzleparameters['ethpercent'] is None:
        rho = 1000 * staticnozzleparameters['pObj'].SG_compressed(temp * const.degKtoR,
                                                                  pres / const.psiToPa)
        return rho
    else:
        rhoeth = 1000 * staticnozzleparameters['pObj'].SG_compressed(temp * const.degKtoR,
                                                                     pres / const.psiToPa)

        rhowater = 1000 * staticnozzleparameters['pObjWater'].SG_compressed(temp * const.degKtoR,
                                                                            pres / const.psiToPa)

        return staticnozzleparameters['ethpercent'] * rhoeth + (1 - staticnozzleparameters['ethpercent']) * rhowater


def cf(temp, pres, fluidvelocity, hydraulicdiameter, staticnozzleparameters, roughness):
    Re = reynolds(temp, pres, fluidvelocity, hydraulicdiameter, staticnozzleparameters['pObj'],
                  staticnozzleparameters['pObjWater'], staticnozzleparameters['ethpercent'])
    if Re < 2100:
        return 16 / Re
    else:
        return scipy.optimize.minimize_scalar(turbulentCfImplicit, args=(roughness, hydraulicdiameter, Re))['x']
    # FIX THIS WHY IS IT GIVING WAY TOO HIGH PRESSURE DROP


def turbulentCfImplicit(cf, roughness, diameter,
                        Re):  # from wikipedia of moody chart lol, we are using fanning friction factor
    try:
        return abs(
            -2 * math.log(roughness / diameter / 3.7 + 2.51 / Re / math.sqrt(cf * 4), 10) - 1 / math.sqrt(cf * 4))
    except:
        return abs(
            -2 * math.log(roughness / diameter / 3.7 + 2.51 / Re / math.sqrt(cf * 4 + .00001), 10) - 1 / math.sqrt(
                (cf + .00001) * 4))


def viscosity(temp, pres, pObj, pObjWater, ethpercent):
    if ethpercent is None:
        return .1 * pObj.Visc_compressed(temp * const.degKtoR,  # Converting from poise to pascal-seconds
                                         pres / const.psiToPa)
    else:

        viscosityeth = .1 * pObj.Visc_compressed(temp * const.degKtoR,
                                                 pres / const.psiToPa)
        viscositywater = .1 * pObjWater.Visc_compressed(temp * const.degKtoR,
                                                        pres / const.psiToPa)

        return ethpercent * viscosityeth + (1 - ethpercent) * viscositywater

def ChanelBoxCorners(xlistoriginal, rlist, twlist, helicitylist, chlist, cwlist):
    ylist = np.zeros(len(xlistoriginal))
    xlist = np.zeros(len(ylist))
    zlist = np.zeros(len(ylist))
    xlist[0] = rlist[0] + twlist[0]
    theta = 0
    for i in range(1, len(ylist)):
        if i % 35 == 0:  # this is just for debugging
            i = i
        if i == 0:
            drdy = (rlist[i + 1] - rlist[i]) / (xlistoriginal[i + 1] - xlistoriginal[i])
        elif i == len(xlistoriginal) - 1:
            drdy = (rlist[i] - rlist[i - 1]) / (xlistoriginal[i] - xlistoriginal[i - 1])
        else:
            drdy = ((rlist[i] - rlist[i - 1]) / (xlistoriginal[i] - xlistoriginal[i - 1]) + (
                        rlist[i + 1] - rlist[i]) / (xlistoriginal[i + 1] - xlistoriginal[i])) / 2
        axialangle = -math.atan(drdy)
        ylist[i] = xlistoriginal[i]
        r = rlist[i] + twlist[i] * math.cos(axialangle)
        # c=math.tan(helicitylist[i])*2*math.pi*r
        dy = (ylist[i] - ylist[i - 1])
        if helicitylist[i] == math.pi / 2:
            dcircum = 0
        else:
            dcircum = dy / math.tan(helicitylist[i])
        dtheta = dcircum / r
        theta = theta + dtheta
        xlist[i] = r * math.cos(theta)
        zlist[i] = r * math.sin(theta)
    # Now we hzve the geometry for the sweep curve, or the curve along rlist+twlist
    # We want to make four lists now, one for each corner of the box
    ylistnew = np.zeros((4, len(xlistoriginal)))
    xlistnew = np.zeros((4, len(ylist)))
    zlistnew = np.zeros((4, len(ylist)))
    # fig = plt.figure() # for showing vectors for debugging
    # ax=fig.add_subplot(projection='3d')
    for i in range(0, len(ylist)):
        angle = abs(math.atan(xlist[i] / zlist[i]))
        # xlist[i]=xlist[i]+math.sin(angle)*np.sign(xlist[i])*chlist[i]
        # zlist[i]=zlist[i]+math.cos(angle)*np.sign(zlist[i])*chlist[i]
        # Find the angle of rlist (converging angle, divering angle, etc)
        if i % 35 == 0:  # this is just for debugging
            i = i
        if i == 0:
            drdy = (rlist[i + 1] - rlist[i]) / (xlistoriginal[i + 1] - xlistoriginal[i])
        elif i == len(xlistoriginal) - 1:
            drdy = (rlist[i] - rlist[i - 1]) / (xlistoriginal[i] - xlistoriginal[i - 1])
        else:
            drdy = ((rlist[i] - rlist[i - 1]) / (xlistoriginal[i] - xlistoriginal[i - 1]) + (
                        rlist[i + 1] - rlist[i]) / (xlistoriginal[i + 1] - xlistoriginal[i])) / 2
        axialangle = -math.atan(drdy)
        # Now find the basis vectors for the plane with the curve as its normal
        # basis1 = [math.sin(angle)*np.sign(xlist[i]), 0 , math.cos(angle)*np.sign(zlist[i])]
        # basis1*basis2 = 0
        # basis2*[0 1 0] = cos(90-rlistangle)*|basis2|
        # basis2 = [A*math.cos(angle)*np.sign(zlist[i]), cos(math.pi/2-rlistangle)*|basis2|, -A*math.sin(angle)*np.sign(xlist[i])]
        # Set |basis2| =1
        # sqrt(A^2+cos(math.pi/2-rlistangle)^2)=1
        # A^2=1-cos(math.pi/2-rlistangle)^2
        Aaxial = math.sqrt(1 - (math.cos(math.pi / 2 - axialangle) ** 2))
        # Ahelicity=math.sqrt(1-(math.cos(helicitylist[i])**2))
        basisheight = [Aaxial * math.sin(angle) * np.sign(xlist[i]), math.cos(math.pi / 2 - axialangle),
                       Aaxial * math.cos(angle) * np.sign(zlist[i])]
        # now with basis height, we want to find a perpendicular vector that is also angled with the helicity
        # for a helicity of 90, the axial (y) component should be zero
        # chaenlcurve vector defined in cylydnircal coordinates : (r,theta,y)
        # -theta/sin(hel)=y/cos(hel)
        # r/sin(axial)=y/cos(axial)
        # r^2+theta^2+y^2=1
        # ytan(hel)^2+ytan(axial)^2+y^2=1
        # y=math.sqrt(1/(1+math.tan(math.pi/2-helicitylist[i])**2+math.tan(axialangle)**2))
        # x = r*sin(angle) +theta*cos(angle)
        # z = r*cos(angle) - theta*sin(agnle)
        # chanelvector = [math.tan(axialangle)*y*math.sin(angle) - math.tan(helicitylist[i])*y*math.cos(angle)
        # , y,
        # math.tan(axialangle)*y*math.cos(angle) + math.tan(helicitylist[i])*y*math.sin(angle)]
        # cylyndrilca basis height(r,theta,z) = (cos(axialangle),0,sin(axialangle))
        # cylidrilca chanel vector = [-tan(axialangle)y, -tan(hel)y, y]
        angle = np.arctan2(xlist[i], zlist[i])  # this is so ham but I did it janky before, now I fix
        if helicitylist[i] == math.pi / 2:
            y = math.sqrt(1 / (1 + math.tan(axialangle) ** 2))
            chanelvector = [-(math.tan(axialangle) * y * math.sin(angle)),
                            y,
                            -(math.tan(axialangle) * y * math.cos(angle))]
        else:
            y = math.sqrt(1 / (1 + (1 / math.tan(helicitylist[i])) ** 2 + math.tan(axialangle) ** 2))
            chanelvector = [
                -(math.tan(axialangle) * y * math.sin(angle) + (1 / math.tan(helicitylist[i])) * y * math.cos(angle)),
                y,
                -(math.tan(axialangle) * y * math.cos(angle) - (1 / math.tan(helicitylist[i])) * y * math.sin(angle))]

        # the perpendicular vector is just chanelvector cross basisheight
        basiswidth = np.cross(chanelvector, basisheight)
        if i == 0:  # this is for the exhaust-side extension (to ensure complete cutting)
            chanelvectorinit = chanelvector
            basisheightinit = basisheight
            basiswidthinit = basiswidth
        # Helps verify vectors point where they should
        # ax.plot([xlist[i],xlist[i] + basiswidth[0]*.01],[ylist[i],ylist[i] + basiswidth[1]*.01],
        #                [zlist[i],zlist[i] + basiswidth[2]*.01],'r')
        # ax.plot([xlist[i],xlist[i] + chanelvector[0]*.01],[ylist[i],ylist[i] + chanelvector[1]*.01],
        #                [zlist[i],zlist[i] + chanelvector[2]*.01],'g')
        # ax.plot([xlist[i],xlist[i] + basisheight[0]*.01],[ylist[i],ylist[i] + basisheight[1]*.01],
        #                [zlist[i],zlist[i] + basisheight[2]*.01],'b')

        # basiswidth = [Ahelicity*math.cos(angle)*np.sign(zlist[i]), math.cos(helicitylist[i]), -Ahelicity*math.sin(angle)*np.sign(xlist[i])]
        xlistnew[0, i] = xlist[i] + basiswidth[0] * cwlist[i] * .5
        ylistnew[0, i] = ylist[i] + basiswidth[1] * cwlist[i] * .5
        zlistnew[0, i] = zlist[i] + basiswidth[2] * cwlist[i] * .5

        xlistnew[1, i] = xlist[i] + basiswidth[0] * cwlist[i] * .5 + basisheight[0] * chlist[i]
        ylistnew[1, i] = ylist[i] + basiswidth[1] * cwlist[i] * .5 + basisheight[1] * chlist[i]
        zlistnew[1, i] = zlist[i] + basiswidth[2] * cwlist[i] * .5 + basisheight[2] * chlist[i]

        xlistnew[2, i] = xlist[i] - basiswidth[0] * cwlist[i] * .5 + basisheight[0] * chlist[i]
        ylistnew[2, i] = ylist[i] - basiswidth[1] * cwlist[i] * .5 + basisheight[1] * chlist[i]
        zlistnew[2, i] = zlist[i] - basiswidth[2] * cwlist[i] * .5 + basisheight[2] * chlist[i]

        xlistnew[3, i] = xlist[i] - basiswidth[0] * cwlist[i] * .5
        ylistnew[3, i] = ylist[i] - basiswidth[1] * cwlist[i] * .5
        zlistnew[3, i] = zlist[i] - basiswidth[2] * cwlist[i] * .5

    # fig.show()

    # Want to extend the list by a bit on both ends so that the cut completes the chanel
    xlistnew = np.concatenate((np.array([[
        xlist[0] - chanelvectorinit[0] * cwlist[0] * 5 + basiswidthinit[0] * cwlist[0] * .5,
        xlist[0] - chanelvectorinit[0] * cwlist[0] * 5 + basiswidthinit[0] * cwlist[0] * .5 + basisheightinit[0] *
        chlist[0],
        xlist[0] - chanelvectorinit[0] * cwlist[0] * 5 - basiswidthinit[0] * cwlist[0] * .5 + basisheightinit[0] *
        chlist[0],
        xlist[0] - chanelvectorinit[0] * cwlist[0] * 5 - basiswidthinit[0] * cwlist[0] * .5
    ]]).T,
                               xlistnew), axis=1)
    ylistnew = np.concatenate((np.array([[
        ylist[0] - chanelvectorinit[1] * cwlist[0] * 5 + basiswidthinit[1] * cwlist[0] * .5,
        ylist[0] - chanelvectorinit[1] * cwlist[0] * 5 + basiswidthinit[1] * cwlist[0] * .5 + basisheightinit[1] *
        chlist[0],
        ylist[0] - chanelvectorinit[1] * cwlist[0] * 5 - basiswidthinit[1] * cwlist[0] * .5 + basisheightinit[1] *
        chlist[0],
        ylist[0] - chanelvectorinit[1] * cwlist[0] * 5 - basiswidthinit[1] * cwlist[0] * .5
    ]]).T,
                               ylistnew), axis=1)
    zlistnew = np.concatenate((np.array([[
        zlist[0] - chanelvectorinit[2] * cwlist[0] * 5 + basiswidthinit[2] * cwlist[0] * .5,
        zlist[0] - chanelvectorinit[2] * cwlist[0] * 5 + basiswidthinit[2] * cwlist[0] * .5 + basisheightinit[2] *
        chlist[0],
        zlist[0] - chanelvectorinit[2] * cwlist[0] * 5 - basiswidthinit[2] * cwlist[0] * .5 + basisheightinit[2] *
        chlist[0],
        zlist[0] - chanelvectorinit[2] * cwlist[0] * 5 - basiswidthinit[2] * cwlist[0] * .5
    ]]).T,
                               zlistnew), axis=1)

    xlistnew = np.concatenate((
        xlistnew,
        np.array([[xlist[i] + chanelvector[0] * cwlist[i] * .5 * 5 + basiswidth[0] * cwlist[i] * .5,
                   xlist[i] + chanelvector[0] * cwlist[i] * .5 * 5 + basiswidth[0] * cwlist[i] * .5 + basisheight[0] *
                   chlist[i],
                   xlist[i] + chanelvector[0] * cwlist[i] * .5 * 5 - basiswidth[0] * cwlist[i] * .5 + basisheight[0] *
                   chlist[i],
                   xlist[i] + chanelvector[0] * cwlist[i] * .5 * 5 - basiswidth[0] * cwlist[i] * .5
                   ]]).T
    ), axis=1)
    ylistnew = np.concatenate((ylistnew,
                               np.array([[
                                   ylist[i] + chanelvector[1] * cwlist[i] * .5 * 5 + basiswidth[1] * cwlist[i] * .5,
                                   ylist[i] + chanelvector[1] * cwlist[i] * .5 * 5 + basiswidth[1] * cwlist[i] * .5 +
                                   basisheight[1] * chlist[i],
                                   ylist[i] + chanelvector[1] * cwlist[i] * .5 * 5 - basiswidth[1] * cwlist[i] * .5 +
                                   basisheight[1] * chlist[i],
                                   ylist[i] + chanelvector[1] * cwlist[i] * .5 * 5 - basiswidth[1] * cwlist[i] * .5
                               ]]).T
                               ), axis=1)
    zlistnew = np.concatenate((
        zlistnew,
        np.array([[
            zlist[i] + chanelvector[2] * cwlist[i] * .5 * 5 + basiswidth[2] * cwlist[i] * .5,
            zlist[i] + chanelvector[2] * cwlist[i] * .5 * 5 + basiswidth[2] * cwlist[i] * .5 + basisheight[2] * chlist[
                i],
            zlist[i] + chanelvector[2] * cwlist[i] * .5 * 5 - basiswidth[2] * cwlist[i] * .5 + basisheight[2] * chlist[
                i],
            zlist[i] + chanelvector[2] * cwlist[i] * .5 * 5 - basiswidth[2] * cwlist[i] * .5
        ]]).T
    ), axis=1)

    return xlistnew, ylistnew, zlistnew


"""#The bean concept is currently as follows:
The temps are calculated assuming box chanels
The bean that best emulates the box is one that is both the same cross sectional area
and the same hydrualic diamemter and the same width. We just have to find the bean
that satisfies all these! We simply solve for all of them being equal with a unique
solution given by adjusting three variables (three unkowns): width, height, and top ellipse vertical axis
The actual bean is normalized to having a width of 1, with the bottom half being the reniform bean
and the top half being an ellipse"""


def ChanelBean(xlistoriginal, rlist, twlist, helicitylist, chlist, cwlist):
    def beanfunc(x, a, b, c, d, s,
                 sign=1):  # here x is in the theta direction (basiswidth), y is in the r direction (basisheight)
        # bean func is x^2 + (y-d)^2 = (a(y-d)^2+bx^2+c(y-d))
        homogenousbean = lambda y: (a * (y * s - d) ** 2 + b * x ** 2 + c * (y * s - d)) ** 2 - x ** 2 - (
                    y * s - d) ** 2
        result = optimize.root(homogenousbean, sign * 10, args=(), method='hybr', jac=None, tol=None, callback=None,
                               options=None)
        return result

    def ellipsefunc(x, axisy, axisx=.5, sign=1):
        return sign * (1 - (x / axisx) ** 2) ** .5 * axisy

    def area(beanpoints):  # beanpoints of the form [0,.1,...,1 ; topleft, ... , topright; bottomleft, ..., bottomright]
        sum = 0
        for i in range(1, beanpoints.shape[1]):
            dx = abs(beanpoints[0, i] - beanpoints[0, i - 1])
            sum = sum + abs(beanpoints[1, i] * dx) + abs(beanpoints[2, i] * dx)
        return sum

    def perimeter(beanpoints):
        sum = beanpoints[1, 0] + beanpoints[1, -1] + beanpoints[2, 0] + beanpoints[2, -1]
        for i in range(1, beanpoints.shape[1]):
            dx = abs(beanpoints[0, i] - beanpoints[0, i - 1])
            sum = sum + np.linalg.norm([beanpoints[1, i] - beanpoints[1, i - 1], dx]) + np.linalg.norm(
                [beanpoints[2, i] - beanpoints[2, i - 1], dx])
        return sum

    def beanmaker(axisyval, ch, beanpoints, a, b, c, d, s, areadesired, cw):
        for index in range(0, beanpoints.shape[1]):
            beanpoints[1, index] = ellipsefunc(beanpoints[0, index], axisy=axisyval, sign=1) * ch
            beanpoints[2, index] = beanfunc(beanpoints[0, index], a, b, c, d, s, sign=-1).x[0] * ch
        beanpoints[0, :] = beanpoints[0, :] * cw
        # plt.plot([cw*.5, cw*.5, -cw*.5, -cw*.5],[areadesired/cw*.5,-areadesired/cw*.5,-areadesired/cw*.5,areadesired/cw*.5,])
        # plt.plot(beanpoints[0,:],beanpoints[1,:])
        ##plt.plot(beanpoints[0,:],beanpoints[2,:])
        # plt.show()
        # return np.linalg.norm([abs(area(beanpoints)-areadesired)/areadesired,abs(perimeter(beanpoints)-perimdesired)/perimdesired])
        return area(beanpoints), perimeter(beanpoints), beanpoints

    ylist = np.zeros(len(xlistoriginal))
    ylist[0] = xlistoriginal[0]
    xlist = np.zeros(len(ylist))
    zlist = np.zeros(len(ylist))
    xlist[0] = rlist[0] + twlist[0]
    theta = 0
    for i in range(1, len(ylist)):
        ylist[i] = xlistoriginal[i]
        r = rlist[i] + twlist[i]
        # c=math.tan(helicitylist[i])*2*math.pi*r
        dy = (ylist[i] - ylist[i - 1])
        if helicitylist[i] == math.pi / 2:
            dcircum = 0
        else:
            dcircum = dy / math.tan(helicitylist[i])
        dtheta = dcircum / r
        theta = theta + dtheta
        xlist[i] = r * math.cos(theta)
        zlist[i] = r * math.sin(theta)
    # Now we hzve the geometry for the sweep curve, or the curve along rlist+twlist
    # We want to make four lists now, one for each corner of the box
    numbeanpoints = 21
    ylistnew = np.zeros((2 * numbeanpoints, len(xlistoriginal)))
    xlistnew = np.zeros((2 * numbeanpoints, len(ylist)))
    zlistnew = np.zeros((2 * numbeanpoints, len(ylist)))
    # fig = plt.figure() # for showing vectors for debugging
    # ax=fig.add_subplot(projection='3d')
    hydraulicdiamlistnew = np.zeros(len(ylist))
    chlistnew = np.zeros(len(ylist))
    for i in range(0, len(ylist)):
        angle = abs(math.atan(xlist[i] / zlist[i]))
        # xlist[i]=xlist[i]+math.sin(angle)*np.sign(xlist[i])*chlist[i]
        # zlist[i]=zlist[i]+math.cos(angle)*np.sign(zlist[i])*chlist[i]
        # Find the angle of rlist (converging angle, divering angle, etc)
        if i % 5 == 0:  # this is just for debugging
            print(i)
        if i == 0:
            drdy = (rlist[i + 1] - rlist[i]) / (xlistoriginal[i + 1] - xlistoriginal[i])
        elif i == len(xlistoriginal) - 1:
            drdy = (rlist[i] - rlist[i - 1]) / (xlistoriginal[i] - xlistoriginal[i - 1])
        else:
            drdy = ((rlist[i] - rlist[i - 1]) / (xlistoriginal[i] - xlistoriginal[i - 1]) + (
                        rlist[i + 1] - rlist[i]) / (xlistoriginal[i + 1] - xlistoriginal[i])) / 2
        axialangle = -math.atan(drdy)
        # Now find the basis vectors for the plane with the curve as its normal
        # basis1 = [math.sin(angle)*np.sign(xlist[i]), 0 , math.cos(angle)*np.sign(zlist[i])]
        # basis1*basis2 = 0
        # basis2*[0 1 0] = cos(90-rlistangle)*|basis2|
        # basis2 = [A*math.cos(angle)*np.sign(zlist[i]), cos(math.pi/2-rlistangle)*|basis2|, -A*math.sin(angle)*np.sign(xlist[i])]
        # Set |basis2| =1
        # sqrt(A^2+cos(math.pi/2-rlistangle)^2)=1
        # A^2=1-cos(math.pi/2-rlistangle)^2
        Aaxial = math.sqrt(1 - (math.cos(math.pi / 2 - axialangle) ** 2))
        # Ahelicity=math.sqrt(1-(math.cos(helicitylist[i])**2))
        basisheight = [Aaxial * math.sin(angle) * np.sign(xlist[i]), math.cos(math.pi / 2 - axialangle),
                       Aaxial * math.cos(angle) * np.sign(zlist[i])]
        # now with basis height, we want to find a perpendicular vector that is also angled with the helicity
        # for a helicity of 90, the axial (y) component should be zero
        # chaenlcurve vector defined in cylydnircal coordinates : (r,theta,y)
        # -theta/sin(hel)=y/cos(hel)
        # r/sin(axial)=y/cos(axial)
        # r^2+theta^2+y^2=1
        # ytan(hel)^2+ytan(axial)^2+y^2=1
        # y=math.sqrt(1/(1+math.tan(math.pi/2-helicitylist[i])**2+math.tan(axialangle)**2))
        # x = r*sin(angle) +theta*cos(angle)
        # z = r*cos(angle) - theta*sin(agnle)
        # chanelvector = [math.tan(axialangle)*y*math.sin(angle) - math.tan(helicitylist[i])*y*math.cos(angle)
        # , y,
        # math.tan(axialangle)*y*math.cos(angle) + math.tan(helicitylist[i])*y*math.sin(angle)]
        # cylyndrilca basis height(r,theta,z) = (cos(axialangle),0,sin(axialangle))
        # cylidrilca chanel vector = [-tan(axialangle)y, -tan(hel)y, y]
        angle = np.arctan2(xlist[i], zlist[i])  # this is so ham but I did it janky before, now I fix
        if helicitylist[i] == math.pi / 2:
            y = math.sqrt(1 / (1 + math.tan(axialangle) ** 2))
            chanelvector = [-(math.tan(axialangle) * y * math.sin(angle)),
                            y,
                            -(math.tan(axialangle) * y * math.cos(angle))]
        else:
            y = math.sqrt(1 / (1 + (1 / math.tan(helicitylist[i])) ** 2 + math.tan(axialangle) ** 2))
            chanelvector = [
                -(math.tan(axialangle) * y * math.sin(angle) + (1 / math.tan(helicitylist[i])) * y * math.cos(angle)),
                y,
                -(math.tan(axialangle) * y * math.cos(angle) - (1 / math.tan(helicitylist[i])) * y * math.sin(angle))]

        # the perpendicular vector is just chanelvector cross basisheight
        basiswidth = np.cross(chanelvector, basisheight)
        if i == 0:  # this is for the exhaust-side extension (to ensure complete cutting)
            chanelvectorinit = chanelvector
            basisheightinit = basisheight
            basiswidthinit = basiswidth
        # Helps verify vectors point where they should
        # ax.plot([xlist[i],xlist[i] + basiswidth[0]*.01],[ylist[i],ylist[i] + basiswidth[1]*.01],
        #                [zlist[i],zlist[i] + basiswidth[2]*.01],'r')
        # ax.plot([xlist[i],xlist[i] + chanelvector[0]*.01],[ylist[i],ylist[i] + chanelvector[1]*.01],
        #                [zlist[i],zlist[i] + chanelvector[2]*.01],'g')
        # ax.plot([xlist[i],xlist[i] + basisheight[0]*.01],[ylist[i],ylist[i] + basisheight[1]*.01],
        #                [zlist[i],zlist[i] + basisheight[2]*.01],'b')

        # basiswidth = [Ahelicity*math.cos(angle)*np.sign(zlist[i]), math.cos(helicitylist[i]), -Ahelicity*math.sin(angle)*np.sign(xlist[i])]

        # numbeanpoints = xlistnew.shape[0]/4 # a bean points defines 2 actual points, one in each half (pos and neg)
        beanpoints = np.linspace(-.45, .45, numbeanpoints)
        beanpoints = np.vstack((beanpoints, np.zeros((2, numbeanpoints))))
        beanpoints[1, :] = ellipsefunc(beanpoints[0, :], axisy=.35, axisx=.5, sign=1) + .5
        beanpoints[2, :] = ellipsefunc(beanpoints[0, :], axisy=.4, axisx=.5, sign=-1) + .5 + .175 * (
                    np.cos(beanpoints[0, :] * math.pi * 2) + 1)
        beanreduce = np.min(beanpoints[2, :])
        beanpoints[1, :] = beanpoints[1, :] - beanreduce
        beanpoints[2, :] = beanpoints[2, :] - beanreduce
        beanmultiply = np.max(beanpoints[1, :])
        beanpoints[1, :] = beanpoints[1, :] / beanmultiply
        beanpoints[2, :] = beanpoints[2, :] / beanmultiply
        beanpoints[1, :] = beanpoints[1, :] * chlist[i]
        beanpoints[2, :] = beanpoints[2, :] * chlist[i]
        beanpoints[0, :] = beanpoints[0, :] * cwlist[i]
        areadesired = chlist[i] * cwlist[i]
        perimdesired = 2 * chlist[i] + 2 * cwlist[i]
        axisyval = .2
        chguess = chlist[i]
        areacurrent = area(beanpoints)
        perim = perimeter(beanpoints)
        tol = .0001
        while abs(areacurrent - areadesired) / areadesired > tol:
            beanpoints = np.linspace(-.45, .45, numbeanpoints)
            beanpoints = np.vstack((beanpoints, np.zeros((2, numbeanpoints))))
            chguess = chguess - (areacurrent - areadesired) / areadesired * chguess
            beanpoints[1, :] = ellipsefunc(beanpoints[0, :], axisy=.35, axisx=.5, sign=1) + .5
            beanpoints[2, :] = ellipsefunc(beanpoints[0, :], axisy=.4, axisx=.5, sign=-1) + .5 + .175 * (
                        np.cos(beanpoints[0, :] * math.pi * 2) + 1)
            beanreduce = np.min(beanpoints[2, :])
            beanpoints[1, :] = beanpoints[1, :] - beanreduce
            beanpoints[2, :] = beanpoints[2, :] - beanreduce
            beanmultiply = np.max(beanpoints[1, :])
            beanpoints[1, :] = beanpoints[1, :] / beanmultiply
            beanpoints[2, :] = beanpoints[2, :] / beanmultiply
            beanpoints = np.vstack((beanpoints, np.zeros((2, numbeanpoints))))
            beanpoints[1, :] = beanpoints[1, :] * chguess
            beanpoints[2, :] = beanpoints[2, :] * chguess
            beanpoints[0, :] = beanpoints[0, :] * cwlist[i]
            chguess = chguess - (areacurrent - areadesired) / areadesired * chguess
            areacurrent = area(beanpoints)
            perim = perimeter(beanpoints)
        # plt.plot(beanpoints[0,:],beanpoints[1,:])
        # plt.plot(beanpoints[0,:],beanpoints[2,:])
        # beanpoints[1,:] = -np.min(beanpoints[2,:])+beanpoints[1,:] #currently bottom is at -.5*ch, gotta shift it up
        # beanpoints[2,:] = -np.min(beanpoints[2,:]) + beanpoints[2,:]
        # plt.plot(beanpoints[0,:],beanpoints[1,:])
        # plt.plot(beanpoints[0,:],beanpoints[2,:])
        # plt.show()
        hydraulicdiamlistnew[i] = 4 * areacurrent / perim
        chlistnew[i] = np.max(beanpoints[1, :])
        if i == 0:  # saving for future extension
            beanpointsinit = beanpoints
        for beanindex in range(0, numbeanpoints):
            xlistnew[beanindex, i] = xlist[i] + basisheight[0] * beanpoints[1, beanindex] + basiswidth[0] * beanpoints[
                0, beanindex]
            ylistnew[beanindex, i] = ylist[i] + basisheight[1] * beanpoints[1, beanindex] + basiswidth[1] * beanpoints[
                0, beanindex]
            zlistnew[beanindex, i] = zlist[i] + basisheight[2] * beanpoints[1, beanindex] + basiswidth[2] * beanpoints[
                0, beanindex]

            xlistnew[numbeanpoints * 2 - beanindex - 1, i] = xlist[i] + basisheight[0] * beanpoints[2, beanindex] + \
                                                             basiswidth[0] * beanpoints[0, beanindex]
            ylistnew[numbeanpoints * 2 - beanindex - 1, i] = ylist[i] + basisheight[1] * beanpoints[2, beanindex] + \
                                                             basiswidth[1] * beanpoints[0, beanindex]
            zlistnew[numbeanpoints * 2 - beanindex - 1, i] = zlist[i] + basisheight[2] * beanpoints[2, beanindex] + \
                                                             basiswidth[2] * beanpoints[0, beanindex]
        print(i)

    # fig.show()

    # Want to extend the list by a bit on both ends so that the cut completes the chanel
    precatarray = np.zeros((numbeanpoints * 2, 1))
    for beanindex in range(0, numbeanpoints):
        precatarray[beanindex, 0] = xlist[0] - chanelvectorinit[0] * cwlist[0] + basisheightinit[0] * beanpointsinit[
            1, beanindex] + basiswidthinit[0] * beanpointsinit[0, beanindex]
        precatarray[numbeanpoints * 2 - beanindex - 1, 0] = xlist[0] - chanelvectorinit[0] * cwlist[0] + \
                                                            basisheightinit[0] * beanpointsinit[2, beanindex] + \
                                                            basiswidthinit[0] * beanpointsinit[0, beanindex]

    xlistnew = np.concatenate((precatarray,
                               xlistnew), axis=1)

    for beanindex in range(0, numbeanpoints):
        precatarray[beanindex, 0] = ylist[0] - chanelvectorinit[1] * cwlist[0] + basisheightinit[1] * beanpointsinit[
            1, beanindex] + basiswidthinit[1] * beanpointsinit[0, beanindex]
        precatarray[numbeanpoints * 2 - beanindex - 1, 0] = ylist[0] - chanelvectorinit[1] * cwlist[0] + \
                                                            basisheightinit[1] * beanpointsinit[2, beanindex] + \
                                                            basiswidthinit[1] * beanpointsinit[0, beanindex]

    ylistnew = np.concatenate((precatarray,
                               ylistnew), axis=1)

    for beanindex in range(0, numbeanpoints):
        precatarray[beanindex, 0] = zlist[0] - chanelvectorinit[2] * cwlist[0] + basisheightinit[2] * beanpointsinit[
            1, beanindex] + basiswidthinit[2] * beanpointsinit[0, beanindex]
        precatarray[numbeanpoints * 2 - beanindex - 1, 0] = zlist[0] - chanelvectorinit[2] * cwlist[0] + \
                                                            basisheightinit[2] * beanpointsinit[2, beanindex] + \
                                                            basiswidthinit[2] * beanpointsinit[0, beanindex]

    zlistnew = np.concatenate((precatarray,
                               zlistnew), axis=1)

    postcatarray = np.zeros((numbeanpoints * 2, 1))
    for beanindex in range(0, numbeanpoints):
        postcatarray[beanindex, 0] = xlist[i] + chanelvector[0] * cwlist[i] + basisheight[0] * beanpoints[
            1, beanindex] + basiswidth[0] * beanpoints[0, beanindex]
        postcatarray[numbeanpoints * 2 - beanindex - 1, 0] = xlist[i] + chanelvector[0] * cwlist[i] + basisheight[0] * \
                                                             beanpoints[2, beanindex] + basiswidth[0] * beanpoints[
                                                                 0, beanindex]

    xlistnew = np.concatenate((
        xlistnew,
        postcatarray
    ), axis=1)

    for beanindex in range(0, numbeanpoints):
        postcatarray[beanindex, 0] = ylist[i] + chanelvector[1] * cwlist[i] + basisheight[1] * beanpoints[
            1, beanindex] + basiswidth[1] * beanpoints[0, beanindex]
        postcatarray[numbeanpoints * 2 - beanindex - 1, 0] = ylist[i] + chanelvector[1] * cwlist[i] + basisheight[1] * \
                                                             beanpoints[2, beanindex] + basiswidth[1] * beanpoints[
                                                                 0, beanindex]

    ylistnew = np.concatenate((
        ylistnew,
        postcatarray
    ), axis=1)

    for beanindex in range(0, numbeanpoints):
        postcatarray[beanindex, 0] = zlist[i] + chanelvector[2] * cwlist[i] + basisheight[2] * beanpoints[
            1, beanindex] + basiswidth[2] * beanpoints[0, beanindex]
        postcatarray[numbeanpoints * 2 - beanindex - 1, 0] = zlist[i] + chanelvector[2] * cwlist[i] + basisheight[2] * \
                                                             beanpoints[2, beanindex] + basiswidth[2] * beanpoints[
                                                                 0, beanindex]

    zlistnew = np.concatenate((
        zlistnew,
        postcatarray
    ), axis=1)

    return xlistnew, ylistnew, zlistnew, hydraulicdiamlistnew, chlistnew


def rlistExtender(xlist, rlist, ewlist):
    externalrlist = np.zeros(len(rlist))
    newxlist = np.zeros(len(xlist))
    normallist = np.arctan2(np.diff(rlist), np.diff(xlist)) + math.pi / 2
    normallist = np.concatenate(([normallist[0]], normallist))
    externalrlist = rlist + np.sin(normallist) * ewlist
    newxlist = xlist + np.cos(normallist) * ewlist
    return newxlist, externalrlist

#This is an old setup for the conventionally manufactured coaxial shell design.
#If this design is ever revisited, the code should be refactored to follow the object oriented
#structure of the other setups. Implment a fin factor function that is 1 everywhere.
"""def coaxialShellSetup(thrustchamber, params, 


    rlist, tw, chanelthickness, helicity, dt, vlist = None, alist = None):
if vlist is None:
    vlist = params['mdot_fuel'] / params['rho_fuel'] / alist
else:
    alist = params['mdot_fuel'] / params['rho_fuel'] / vlist
n = 1

twlist = tw * np.ones((1, rlist.size))
vInterpolator = interpolate.interp1d(thrustchamber.xlist,
                                     vlist, kind='linear')
rInterpolator = interpolate.interp1d(thrustchamber.xlist,
                                     rlist, kind='linear')
twInterpolator = interpolate.interp1d(thrustchamber.xlist,
                                      twlist, kind='linear')
aInterpolator = interpolate.interp1d(thrustchamber.xlist,
                                     alist, kind='linear')
chanelthicknessInterpolator = interpolate.interp1d(thrustchamber.xlist,
                                                   chanelthickness, kind='linear')
x = thrustchamber.xlist[-1]
ind = 0
# xlist = np.zeros(int(thrustchamber.xlist[-1] / dt / np.min(vlist)))
# dxlist = np.zeros(int(thrustchamber.xlist[-1] / dt / np.min(vlist)))
xlist = np.zeros(int(thrustchamber.xlist[-1] / dt) + 1)
dxlist = np.zeros(int(thrustchamber.xlist[-1] / dt) + 1)

xlist[0] = x
dx = dt
while x > 0 + dx * 1.01:

    dydx = (rInterpolator(x) - rInterpolator(x - dx)) / dx
    dhypotenuse = vInterpolator(x) * dt
    dx = dt  # (dhypotenuse ** 2 / ((dydx ** -2) + 1) / (dydx ** 2)) ** .5
    if dydx == 0:
        dx = dt  # dhypotenuse

    dxlist[ind] = dx
    x = xlist[ind] - dx
xlist[ind + 1] = x
ind = ind + 1

whilexlist[-1] == 0
andxlist[-2] == 0:
xlist = xlist[0:-1]
dxlist = dxlist[0:-1]  # trims the extra zeros
dxlist[-1] = dxlist[-2]
alistflipped = aInterpolator(xlist)
vlistflipped = vInterpolator(xlist)
xlistflipped = xlist
twlistflipped = twInterpolator(xlist)[0]
salistflipped = math.pi * 2 * (rInterpolator(xlist) + twlistflipped) * dxlist
hydraulicdiamlist = 2 * (
    chanelthicknessInterpolator(xlist))  # hydraulic diam is Douter-Dinner for an anulus by wetted perimiter
coolingfactorlist = np.ones(xlist.size)
heatingfactorlist = np.ones(xlist.size)  # .6 is from cfd last year, i think its bs but whatever

return alistflipped, n, coolingfactorlist, heatingfactorlist, xlistflipped, vlistflipped, twlistflipped, hydraulicdiamlist, salistflipped"""