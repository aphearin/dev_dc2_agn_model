"""The `calculate_quasar_probability` function implements a model for
the probability that a galaxy hosts a quasar.

The `calculate_quasar_dimming` function implements a model for
the dimming of quasars below the brightness of the existing cosmoDC2 implementation.
"""
import numpy as np


def calculate_quasar_probability(stellar_mass, redshift, gr_restframe, ri_restframe,
            hiz_main_sequence_prob_boost=0.1, **kwargs):
    """ Model for the probability that an object hosts a quasar based
    on the distance from the green valley. Galaxies with colors that are
    redder than the green valley have a very low probability of hosting a quasar.
    Galaxies with bluer colors have a quasar probability that is tunable by
    the input model parameters.

    An explicit redshift-dependence of the model can be tuned by the input
    `hiz_main_sequence_prob_boost` parameter; note that even when hiz_prob_boost = 0,
    quasars will still be more abundant at higher redshift because
    the blue fraction of galaxies is larger at higher redshift.

    Parameters
    ----------
    stellar_mass : float or ndarray of shape (ngals, )

    redshift : float or ndarray of shape (ngals, )

    gr_restframe : float or ndarray of shape (ngals, )

    ri_restframe : float or ndarray of shape (ngals, )

    hiz_main_sequence_prob_boost : float, optional
        Increased probability at high-redshift for a main sequence galaxy
        will host a quasar relative to a z=0 galaxy with the same mass and colors.

    Returns
    -------
    quasar_prob : ndarray of shape (ngals, )
    """
    stellar_mass, redshift, gr_restframe, ri_restframe = _get_1d_arrays(
                stellar_mass, redshift, gr_restframe, ri_restframe)
    eigencolor = calculate_eigencolor(gr_restframe, ri_restframe)
    gv_distance = _calculate_gv_distance(stellar_mass, eigencolor, **kwargs)
    quasar_prob = _quasar_prob_vs_gv_distance(gv_distance, redshift,
            hiz_main_sequence_prob_boost=hiz_main_sequence_prob_boost, **kwargs)

    quasar_prob = np.where(quasar_prob < 0, 0, quasar_prob)
    quasar_prob = np.where(quasar_prob > 1, 1, quasar_prob)
    return quasar_prob


def calculate_quasar_dimming(stellar_mass, redshift, delta_mag_bcgs_z0=1, delta_mag_bcgs_z2=2,
            delta_mag_dwarf_z0=0., delta_mag_dwarf_z2=0., logsm_char=10.5, **kwargs):
    """Calculate the change in magnitude that should be applied to the quasar,
    so that positive values of the returned function correspond to fainter quasars.
    The value of the function can be specified by model parameters for low-mass
    "dwarf" galaxies, and high-mass "bcg" galaxies, separately at low- and high-redshift,
    using sigmoid interpolation in both stellar mass and redshift.

    Parameters
    ----------
    stellar_mass : float or ndarray of shape (ngals, )

    redshift : float or ndarray of shape (ngals, )

    Returns
    -------
    delta_mag : ndarray of shape (ngals, )
    """
    logsm = np.log10(_get_1d_arrays(stellar_mass))
    delta_mag_dwarf = _sigmoid(
            redshift, x0=1, k=3, ylo=delta_mag_dwarf_z0, yhi=delta_mag_dwarf_z2)
    delta_mag_bcg = _sigmoid(
            redshift, x0=1, k=3, ylo=delta_mag_bcgs_z0, yhi=delta_mag_bcgs_z2)
    return _sigmoid(logsm, x0=logsm_char, k=2, ylo=delta_mag_dwarf, yhi=delta_mag_bcg)


def calculate_eigencolor(gr_restframe, ri_restframe, gr0=0.93, ri0=0.37, **kwargs):
    """ Project g-r and r-i onto a direction roughly pointing along the first
    principal component seen in SDSS data at low redshift.

    Parameters
    ----------
    gr_restframe : float or ndarray of shape (ngals, )

    ri_restframe : float or ndarray of shape (ngals, )

    Returns
    -------
    eigencolor : ndarray of shape (ngals, )
    """
    evec = np.array((gr0, ri0))
    normed_evec = evec/np.sqrt(np.sum(evec**2))
    colors = np.vstack((gr_restframe, ri_restframe)).T
    eigencolor = np.dot(colors, np.reshape(normed_evec, (normed_evec.size, 1)))
    return eigencolor.flatten()


def eigencolor_green_valley(stellar_mass, logsm0=10., y0=0.8, color_vs_logsm_slope=0.15, **kwargs):
    """Location of the bottom of the green valley as a function of stellar mass.
    The green valley is defined in terms of an eigencolor,
    a linear combination of g-r and r-i. The eigencolor_green_valley function defines
    how the eigencolor should be split into "blue" and "red" samples based on the
    M*-dependent location of the green valley in the eigencolor.

    Parameters
    ----------
    stellar_mass : float or ndarray of shape (ngals, )

    Returns
    -------
    eigencolor : ndarray of shape (ngals, )
        Location of the green valley in the eigencolor
    """
    return (np.log10(stellar_mass) - logsm0)*color_vs_logsm_slope + y0


def _quasar_prob_vs_gv_distance(gv_distance, redshift, hiz_main_sequence_prob_boost=0.1,
            main_sequence_logprob_z0=-0.6, quenched_logprob=-3, **kwargs):
    """Calculate the probability that a galaxy hosts a quasar as a function of DV,
    the distance from the Green Valley. Galaxies with positive GV have a red
    eigencolor and are very unlikely to host a quasar; galaxies with negative GV
    have a blue eigencolor and host a quasar with probability tunable by the
    `main_sequence_logprob` and `hiz_main_sequence_prob_boost` model parameters.

    """
    prob_hiz_boost = _sigmoid(redshift, x0=0.5, k=4, ylo=0, yhi=hiz_main_sequence_prob_boost)
    main_sequence_prob_z0 = 10.**main_sequence_logprob_z0
    main_sequence_prob = main_sequence_prob_z0 + prob_hiz_boost
    main_sequence_logprob = np.log10(main_sequence_prob)
    return 10**_sigmoid(
                gv_distance, x0=0, k=5, ylo=main_sequence_logprob, yhi=quenched_logprob)


def _calculate_gv_distance(mstar, eigencolor, **kwargs):
    gv_loc = eigencolor_green_valley(mstar, **kwargs)
    gv_dist = eigencolor - gv_loc
    return gv_dist


def _sigmoid(x, x0, k, ylo, yhi):
    height_diff = yhi-ylo
    return ylo + height_diff/(1 + np.exp(-k*(x-x0)))


def _get_1d_arrays(*args):
    """Return ndarrays of equal length, or raise an exception if not possible
    """
    results = [np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)
    return [np.zeros(npts).astype(arr.dtype) + arr for arr in results]
