# this is for reading in csv 

# first, the import statemtnts
import numpy, scipy.signal
import matplotlib.pyplot as plt

def savitzky_golay( y, window_size, order, deriv = 0 ):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = numpy.linspace(-4, 4, 500)
    y = numpy.exp( -t**2 ) + numpy.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, numpy.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = numpy.abs( numpy.int( window_size ) )
        order = numpy.abs( numpy.int( order ) )
    except ValueError, msg:
        raise ValueError( "window_size and order have to be of type int" )
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError( "window_size size must be a positive odd number" )
    if window_size < order + 2:
        raise TypeError( "window_size is too small for the polynomials order" )
    order_range = range( order + 1 )
    half_window = ( window_size - 1 ) // 2
    # precompute coefficients
    b = numpy.mat( [[k ** i for i in order_range] for k in range( -half_window, half_window + 1 )] )
    m = numpy.linalg.pinv( b ).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - numpy.abs( y[1:half_window + 1][::-1] - y[0] )
    lastvals = y[-1] + numpy.abs( y[-half_window - 1:-1][::-1] - y[-1] )
    y = numpy.concatenate( ( firstvals, y, lastvals ) )
    return numpy.convolve( m, y, mode = 'valid' )


# taken from the old thing, just a definition of the file and headerlines
fileName = "I:\NewFile2.csv"		# enter the file name, bitch
headLines = 3		

# we first acquire the data. In this case the extra 'z' is given because there was a 2 channel oscilloscope, one channel of which was OFF.
t, chanA, chanB = numpy.genfromtxt(fileName, skip_header=headLines, unpack=True, delimiter=',')

# first, a filter
# we start by finding the upper and lower frequency bounds of this data...
fs = 1/(t[1]-t[0])			# sampling rate, -ve taken because its negative for some reason
fnyq = fs/2					# nyquist frequency

print "sampling frequency =", fs
print "lowest frequency =", 1/(t[-1]-t[0])

# knowing these, we can start experimenting with a bandpass filter that would be most to our satisfaction..
# variables relating to the filter:
fCutoffHigh = 8	# Hz
butterOrder = 3

# filter definition... we start with the lowpass filter
b, a = scipy.signal.butter(butterOrder, [fCutoffHigh/fnyq])

# next we apply the filter on the 'chanA' variable (which is the data)
filteredAmpl = scipy.signal.lfilter(b, a, chanA)

# then we smooth it...
smoothedChanA = savitzky_golay(filteredAmpl, window_size=27, order=4)

# now we plot the files
plt.plot(t, chanA, 'k', label='Original signal')
plt.plot( t, filteredAmpl, 'r', label='Filtered LP 5Hz 3rd order signal')
plt.plot( t, smoothedChanA, 'b', label='Savitzky-Golayed signal')
plt.axis([-6,6,0,1.2])
plt.grid(True)

plt.xlabel('time (seconds)')
plt.legend(loc='lower right')

plt.show()		# don't forget this!
