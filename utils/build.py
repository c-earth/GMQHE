#########################################################
# helper file for building the specimens for simulations
#########################################################

def create_platform(model, shape, center, dim):
    '''
    Create plain kwant object with specific shape, model.structure.lattice parameters,
    and specific hopping Hamiltonian (model.H)
    Arguments:
        model   :   (utils.dataprep.TB_model)
                    tight-binding model of the material
        shape   :   (boolean function)
                    function indicating whether a given point is in the shape or not
        center  :   (numpy.array)
                    a point that is inside the shape
        dim     :   (int)
                    dimension of the specimen
    Output:
        (kwant)
    '''
    return

def etching(specimen, shape, dim):
    '''
    Modify a specimen by etching some sites out according to the shape given
    Arguments:
        specimen    :   (kwant)
        shape       :   (boolean function)
                        indicating whether a given point is in the shape or not
        dim         :   (int)
                        dimension of the etching
    Output:
        (kwant)
    '''
    return

def create_specimen(model, shape, center, dim, etchings):
    '''
    Create the specimen ready for simulation
    Arguments:
        model   :   (utils.dataprep.TB_model)
                    tight-binding model of the material
        shape   :   (boolean function)
                    indicating whether a given point is in the shape or not
        center  :   (numpy.array)
                    a point that is inside the shape
        dim     :   (int)
                    dimension of the specimen
        etchings:   (list of lists)
                    shapes and dimensions of every etching to create the specimen
    Output:
        (kwant)
    '''
    return

def create_specimens(orders):
    '''
    Create specimens for each of the set of parameters given in orders
    Arguments:
        orders  :   (dict)
                    order_name (str) -> all parameters needed to create a specimen (list)
    Output:
        (dict)  :   order_name (str) -> specimen (kwant)
    '''
    return

def lead_attaching(specimens, lead_orders):
    '''
    Attached leads to each specimen according to lead_orders
    '''
    return