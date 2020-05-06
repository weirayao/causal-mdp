import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.cm as cm
from celluloid import Camera

def visualizer(X, history=False, t_range=3):
    '''
    Visualizes the balls in the environment.

            Parameters:
                    X (numpy.ndarray): Input of shape (time_steps, objects, features)
                    history (bool): Want historical traces or not
                    t_range (int): Number of time steps back we want to see

            Returns:
                    animation (matplotlib.animation.ArtistAnimation): Animation object which can be displayed/saved
    '''
    
    assert t_range >= 0
    
    if t_range == 0:
        history=False
    
    # setting the context
    lim_x_left = np.ravel(X[:, :, 2]).min()
    lim_x_right = np.ravel(X[:, :, 4]).max()
    lim_y_bottom = np.ravel(X[:, :, 3]).min()
    lim_y_top = np.ravel(X[:, :, 5]).max()
    fig, ax = plt.subplots()
    plt.axis('off')
    camera = Camera(fig)
    ax.set_xlim((lim_x_left, lim_x_right))
    ax.set_ylim((lim_y_bottom, lim_y_top))
    ax.set_aspect("equal")

    # setting up the agents
    duration, n_balls, _ = X[:,4:,:].shape
    
    radii = []
    for ball in range(n_balls): # look to the first ball to get the radius
        radius = X[0, 4+ball, 2:6]
        radius = 0.25 * (radius[2] - radius[0] + radius[3] - radius[1])
        radii.append(radius)
    
    if history:
        start = t_range + 1
    else:
        start = 0
        
    # drawing the balls, start from time 5
    for t in range(start, duration):
        
        # draw the walls
        ax.axvline(x=lim_x_left, c='black')
        ax.axvline(x=lim_x_right, c='black')
        ax.axhline(y=lim_y_top, c='black')
        ax.axhline(y=lim_y_bottom, c='black')
        
        colors = iter(cm.rainbow(np.linspace(0, 1, n_balls)))
        
        for ball in range(n_balls):

            pos = X[t, 4+ball, [2, 3]]
            radius = radii[ball]
            color = next(colors)
            c = plt.Circle((pos[0]+radius, pos[1]+radius), radius, color=color, clip_on=False)

            ax.add_artist(c)
            
            if history:
                
                for his in range(t_range):
                    
                    pos = X[t-1-his, 4+ball, [2, 3]]
                    radius = radii[ball]
                    color[3] = 1. - ((his+1)/(t_range+1)) # adding fade
                    c = plt.Circle((pos[0]+radius, pos[1]+radius), radius, color=color, clip_on=False)

                    ax.add_artist(c)

        camera.snap()
    
    animation = camera.animate()
    
    return animation