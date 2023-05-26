#Fix this
class Component(object):
    upstream = None
    downstream = None
    def Step(self, dt : float , args : dict):
        #update the components state based on upstream and downstream
        # downstream should have a reference to this object
        pass

    def Equilibrium(self, args : dict):
        #solve for equilibrium state of component
        pass
