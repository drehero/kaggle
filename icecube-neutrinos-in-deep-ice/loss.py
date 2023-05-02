import torch


def vMF_loss(n_pred, n_true, eps = 1e-8):
    # https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/discussion/392096
    """  von Mises-Fisher Loss: n_true is unit vector ! """
    kappa = torch.norm(n_pred, dim=1)
    logC  = -kappa + torch.log( ( kappa+eps )/( 1-torch.exp(-2*kappa)+2*eps ) )
    return -( (n_true*n_pred).sum(dim=1) + logC ).mean()


def angular_dist_score(az_pred, zen_pred, az_true, zen_true):
    '''
    calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse
    cosine (arccos) thereof is then the angle between the two input vectors

    Parameters:
    -----------

    az_pred : float (or array thereof)
        predicted azimuth value(s) in radian
    zen_pred : float (or array thereof)
        predicted zenith value(s) in radian
    az_true : float (or array thereof)
        true azimuth value(s) in radian
    zen_true : float (or array thereof)
        true zenith value(s) in radian

    Returns:
    --------

    dist : float
        mean over the angular distance(s) in radian
    '''

    # pre-compute all sine and cosine values
    sa1 = torch.sin(az_true)
    ca1 = torch.cos(az_true)
    sz1 = torch.sin(zen_true)
    cz1 = torch.cos(zen_true)

    sa2 = torch.sin(az_pred)
    ca2 = torch.cos(az_pred)
    sz2 = torch.sin(zen_pred)
    cz2 = torch.cos(zen_pred)

    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1*sz2*(ca1*ca2 + sa1*sa2) + (cz1*cz2)

    # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # that might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod = torch.clamp(scalar_prod, -1, 1)

    # convert back to an angle (in radian)
    return torch.mean(torch.abs(torch.arccos(scalar_prod)))
