import numpy as np
import scipy.sparse as sp
import numpy.linalg as npl
import scipy.special as ss
import scipy.optimize as opt
from scipy import interpolate
import matplotlib.pyplot as plt
from ipywidgets import interact_manual

import warnings
warnings.filterwarnings("ignore")

# **** Comment the 3 following lines if it raises an error. ****
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('font', size = 20)

#plt.style.use('science')


overlap = np.load('overlap.npy')
overlap[0] = np.flip(overlap[0])
overlap[1] = np.flip(overlap[1])

overlap[1, 45: 48] = 0.91, 0.93, 0.95

def moving_average(x, y, degree):
    smooth_x, smooth_y = np.copy(x), np.zeros(x.shape[0])
    smooth_y[0] = np.mean(y[:2])
    smooth_y[-1] = np.mean(y[-2:])
    for i in range(1, smooth_y.shape[0] - 1):
        if i < degree//2:
            smooth_y[i] = np.mean(y[: 2*i + 1])
        elif i > smooth_x.shape[0] - degree//2:
            smooth_y[i] = np.mean(y[i - (smooth_y.shape[0] - i - 1):])
        else:
            smooth_y[i] = np.mean(y[i - degree//2: i + degree//2 + 1])     
    return smooth_x, smooth_y

def shorting(x, y, modulo):
    x_short, y_short = [], []
    for i in range(x.shape[0]):
        if i % modulo == 0:
            x_short += [x[i]]
            y_short += [y[i]]
    if x_short[-1] != x[-1]:
        x_short += [x[-1]]
        y_short += [y[-1]]
    return np.array(x_short), np.array(y_short)

def smoothing(x, y, degree, modulo):
    x_short, y_short = shorting(x, y, modulo)
    x_average, y_average = moving_average(x, y, degree)
    spline = interpolate.CubicSpline(x_average, y_average)
    new_x = np.linspace(x_average[0], x_average[-1], x.shape[0])
    #return new_x, spline(new_x)
    return x_average, y_average
    

degree = 31
modulo = 5
smooth_overlap = np.array(smoothing(overlap[0], overlap[1], degree, modulo))


def intersection_distance(r_0, r_1, d):
    return (r_0**2 - r_1**2 + d**2) / (2*d)

def intersection_angle(r_0, r_1, d):
    return np.arccos(intersection_distance(r_0, r_1, d) / r_0)

def chi(r_0, r_1, d):
    return ((r_0 + r_1)**2 - d**2)*(d**2 - (r_0 - r_1)**2)

def intersection_area(r_0, r_1, d):    
    if d > r_0 + r_1:
        return 0
    elif d < np.abs(r_0 - r_1):
        return np.pi * np.minimum(r_0**2, r_1**2) 
    elif (d <= r_0 + r_1) * (d >= np.abs(r_0 - r_1)):
        varphi_0, varphi_1 = intersection_angle(r_0, r_1, d), intersection_angle(r_1, r_0, d)
        return varphi_0*r_0**2 + varphi_1*r_1**2 - 0.5*np.sqrt(chi(r_0, r_1, d))    

def rectangular_integration_1d(x, y):
    if x.shape[0] < 2:
        return 0
    else:
        integral = 0
        dx = x[1] - x[0]
        for i in range(x.shape[0]-1):
            integral += y[i+1] + y[i]
        return 0.5 * dx * integral

def rectangular_integration_2d(x, y, matrix):
    if (matrix.shape[0] < 2) + (matrix.shape[1] < 2):
        return 0
    else:
        integral = 0
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        for i in range(x.shape[0]-1):
            for j in range(y.shape[0]-1):
                integral += matrix[i, j] + matrix[i+1, j] + matrix[i, j+1] + matrix[i+1, j+1]
        return 0.25 * dx * dy * integral
    
def normal_monte_carlo_integration(function, law, z, iterations_number):
    integral = 0
    for k in range(iterations_number):
        x = np.random.normal(law.mean[0], np.sqrt(law.variance[0]))
        y = np.random.normal(law.mean[1], np.sqrt(law.variance[1]))
        integral += function(x, y, z)
    return integral / iterations_number

def circular_uniform_monte_carlo_integration(function, law, z, iterations_number):
    integral = 0
    mean = 0.5 * np.array([law.maximum[0] + law.minimum[0], law.maximum[1] + law.minimum[1]])
    for k in range(iterations_number):
        x = np.random.uniform(law.minimum[0], law.maximum[0])
        y = np.random.uniform(law.minimum[1], law.maximum[1])
        while np.sqrt((x - mean[0])**2 + (y - mean[1])**2) > law.maximum[0] - mean[0]:
            x = np.random.uniform(law.minimum[0], law.maximum[0])
            y = np.random.uniform(law.minimum[1], law.maximum[1])
        integral += function(x, y, z)
    return integral / iterations_number
    
def monte_carlo_integration(function, law, z, iterations_number):
    if type(law) == normal_law_object:
        integral = normal_monte_carlo_integration(function, law, z, iterations_number)
    elif type(law) == circular_uniform_law_object:
        integral = circular_uniform_monte_carlo_integration(function, law, z, iterations_number)
    return integral

def bessel_function(x):
    boundary_argument = 709.78
    result = np.zeros(x.shape[0])
    for k in range(result.shape[0]):
        if x[k] < boundary_argument:
            result[k] = ss.i0(x[k])
        else:
            result[k] = ss.i0(boundary_argument)
    return result

def analytic_method(setup, z):
    separation_z = npl.norm(setup.laser.beam_center(z), 2)
    if setup.laser.law_name == "circular_uniform_law":
        laser_radius_z = setup.laser.beam_width(z)
        return intersection_area(setup.telescope.beam_width(z), laser_radius_z, separation_z) / (np.pi * laser_radius_z**2)
    elif setup.laser.law_name == "normal_law":
        laser_law = setup.laser.law_object_creation(z)
        x = np.linspace(0, setup.telescope.beam_width(z), 100)
        y = x * np.exp(-(x**2 + separation_z**2) / (2 * laser_law.variance[0])) * bessel_function(separation_z * x / laser_law.variance[0]) / laser_law.variance[0]
        return rectangular_integration_1d(x, y)
    
class telescope_object:        
    def __init__(self, radius, angle, focal_length):
        self.radius = radius
        self.angle = angle
        self.focal_length = focal_length
        
    def update_radius(self, radius):
        self.radius = radius
        
    def updtate_angle(self, angle):
        self.angle = angle
        
    def update_focal_length(self, focal_length):
        self.focal_length = focal_length
        
    def beam_width(self, z):
        return self.radius + np.tan(self.angle)*z
    
    def flow_point(self, x, y, z):
        return 1 * (x**2 + y**2 <= self.beam_width(z)**2)
    
    def flow_matrix(self, x, y, z):
        matrix = np.zeros((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                matrix[i, j] = self.flow_point(x[i], y[j], z)
        return matrix
    
class circular_uniform_law_object:
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum
        
    def update_minimum(self, minimum):
        self.minimum = minimum
        
    def update_maximum(self, maximum):
        self.maximum = maximum
        
    def density_function(self, x, y):
        c_x = 0.5 * (self.maximum[0] + self.minimum[0])
        c_y = 0.5 * (self.maximum[1] + self.minimum[1])
        radius = c_x - self.minimum[0]
        radius_prime = np.sqrt((x - c_x)**2 + (y - c_y)**2)
        theta = 0.5 * (1 + (radius - radius_prime) / np.abs(radius - radius_prime))
        return theta / (np.pi*radius**2)
    
    def density_matrix(self, x, y):
        matrix = np.zeros((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                matrix[i, j] = self.density_function(x[i], y[j])
        return matrix
    
class normal_law_object:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance
        
    def update_mean(self, mean):
        self.mean = mean
        
    def update_variance(self, variance):
        self.variance = variance
        
    def density_function(self, x, y):
        f_x = np.exp(-(x - self.mean[0])**2 / (2*self.variance[0])) / (np.sqrt(2*np.pi*self.variance[0]))
        f_y = np.exp(-(y - self.mean[1])**2 / (2*self.variance[1])) / (np.sqrt(2*np.pi*self.variance[1]))
        return f_x * f_y
    
    def density_matrix(self, x, y):
        matrix = np.zeros((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                matrix[i, j] = self.density_function(x[i], y[j])
        return matrix
    
class laser_object:
    def __init__(self, radius, angle, center, alpha, beta, law_name):
        self.radius = radius
        self.angle = angle
        self.center = center
        self.alpha = alpha
        self.beta = beta
        self.law_name = law_name
        
    def update_radius(self, radius):
        self.radius = radius
        
    def update_angle(self, angle):
        self.angle = angle        
        
    def update_center(self, center):
        self.center = center
        
    def update_alpha(self, alpha):
        self.alpha = alpha
    
    def update_beta(self, beta):
        self.beta = beta

    def update_law_name(self, law_name):
        self.law_name = law_name
        
    def beam_center(self, z):
        return self.center - z * np.tan(self.alpha) * np.array([np.cos(self.beta), -np.sin(self.beta)])
    
    def beam_width(self, z):
        distance = z / np.cos(self.alpha)
        if self.law_name == "normal_law":
            return np.sqrt(self.radius**2 + (self.angle*distance)**2)
        elif self.law_name == "circular_uniform_law":
            return self.radius + self.angle*distance
    
    def law_object_creation(self, z):
        if self.law_name == "circular_uniform_law":
            laser_radius_z = self.beam_width(z)
            minimum = self.beam_center(z) - laser_radius_z * np.ones(2)
            maximum = self.beam_center(z) + laser_radius_z * np.ones(2)
            return circular_uniform_law_object(minimum, maximum)
        elif self.law_name == "normal_law":
            laser_radius_z = self.beam_width(z)
            mean = self.beam_center(z)
            variance = (0.5 * laser_radius_z)**2 * np.ones(2)
            return normal_law_object(mean, variance)
        
class laser_telescope_setup_object:
    def __init__(self, telescope, laser, separation, method):
        self.telescope = telescope
        self.laser = laser
        self.separation = separation
        self.method = method
        
    def beams_separation(self, z):
        return np.sqrt(np.sum(self.laser.beam_center(z)**2))
        
    def gff_integrand_point(self, x, y, z):
        law = self.laser.law_object_creation(z)
        return law.density_function(x, y) * self.telescope.flow_point(x, y, z)
    
    def gff_integrand_matrix(self, x, y, z):
        matrix = np.zeros((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                matrix[i, j] = self.gff_integrand_point(x[i], y[j], z)
        return matrix
    
    def space_restriction(self, x, y, z):
        index = np.arange(0, x.shape[0], 1)
        telescope_radius_z = self.telescope.beam_width(z)
        laser_radius_z = self.laser.beam_width(z)
        separation_z = self.beams_separation(z)
        c_x, c_y = self.laser.beam_center(z)
        if separation_z >= laser_radius_z + telescope_radius_z:
            x_z = np.array([])
            y_z = np.array([])
        else:
            if self.laser.law_name == "circular_uniform_law":
                if separation_z <= np.abs(laser_radius_z - telescope_radius_z):
                    x_z = x[index[(x > c_x - laser_radius_z)*(x < c_x + laser_radius_z)]]
                    y_z = y[index[(y > c_y - laser_radius_z)*(y < c_y + laser_radius_z)]]
                else:
                    if laser_radius_z > separation_z:
                        y_z = y[index[(y > -telescope_radius_z)*(y < telescope_radius_z)]]
                    else:
                        alpha = np.arccos(np.sqrt(1 - laser_radius_z**2/separation_z**2))
                        y_z = y[index[(y > -telescope_radius_z*np.sin(alpha)) * (y < telescope_radius_z*np.sin(alpha))]]
                    x_z = x[index[(x > c_x - laser_radius_z)*(x < telescope_radius_z)]]
            elif self.laser.law_name == "normal_law":
                x_z = x[index[(x > -telescope_radius_z)*(x < telescope_radius_z)]]
                y_z = y[index[(y > -telescope_radius_z)*(y < telescope_radius_z)]]
        return x_z, y_z
    
    def gff_calculation_fixed_distance(self, x, y, z, iterations_number = 0):
        if self.method == rectangular_integration_2d:
            x, y = self.space_restriction(x, y, z)
            return rectangular_integration_2d(x, y, self.gff_integrand_matrix(x, y, z))
        elif self.method == monte_carlo_integration:
            laser_law = self.laser.law_object_creation(z)
            return monte_carlo_integration(self.telescope.flow_point, laser_law, z, iterations_number)
        elif self.method == analytic_method:
            return analytic_method(self, z)
        
    def gff_calculation_unfixed_distance(self, x, y, z, iterations_number = 0):
        gff_list = np.zeros(z.shape[0])
        for k in range(z.shape[0]):
            gff_list[k] = self.gff_calculation_fixed_distance(x, y, z[k], iterations_number)
        return gff_list
   
def interactive_gff_display(Tlscp_radius, Tlscp_angle, Laser_radius, Laser_angle, Separation, Alpha, Beta, Analytic, Monte_Carlo, Z_min, Z_max, Z_nb, Setup_at_z):  
    figsize = (8, 7)
    
    telescope_radius, telescope_angle = float(Tlscp_radius), float(Tlscp_angle)
    laser_radius, laser_angle = float(Laser_radius), float(Laser_angle)
    separation = float(Separation)
    alpha, beta = float(Alpha), float(Beta)
    focal_length = 1
    
    z_min, z_max = float(Z_min), float(Z_max)
    z_nb = int(Z_nb)
    
    setup_at_z = float(Setup_at_z)
    
    laser_center = np.array([separation, 0])

    telescope = telescope_object(telescope_radius, telescope_angle, focal_length)
    laser_1 = laser_object(laser_radius, laser_angle, laser_center, alpha, beta, "circular_uniform_law")
    laser_2 = laser_object(laser_radius, laser_angle, laser_center, alpha, beta, "normal_law")        
    
    z = np.linspace(z_min, z_max, z_nb)
    
    mesh_nb = 250
    maximum_telescope_radius = telescope.beam_width(z[-1])
    x = np.linspace(-maximum_telescope_radius, maximum_telescope_radius, mesh_nb)
    y = np.linspace(-maximum_telescope_radius, maximum_telescope_radius, mesh_nb)
    
    if Monte_Carlo:
        iterations_number = 1000
        
        setup_12 = laser_telescope_setup_object(telescope, laser_1, separation, monte_carlo_integration)
        setup_22 = laser_telescope_setup_object(telescope, laser_2, separation, monte_carlo_integration)
        
        gff_12 = setup_12.gff_calculation_unfixed_distance(x, y, z, iterations_number)
        gff_22 = setup_22.gff_calculation_unfixed_distance(x, y, z, iterations_number)
        
        plt.figure(figsize = figsize)
        plt.plot(z, gff_12, '--', color = "C2", label = "(a)")
        plt.plot(z, gff_22, '--', color = "C1", label = "(b)") 
    
        if Analytic:
            setup_11 = laser_telescope_setup_object(telescope, laser_1, separation, analytic_method)
            setup_21 = laser_telescope_setup_object(telescope, laser_2, separation, analytic_method)
            
            gff_11 = setup_11.gff_calculation_unfixed_distance(x, y, z)
            gff_21 = setup_21.gff_calculation_unfixed_distance(x, y, z)
            
            plt.plot(z, gff_11, color = "r", linewidth = 2, label = "(c)")
            plt.plot(z, gff_21, color = "C0", linewidth = 2, label = "(d)")
    else:
        setup_11 = laser_telescope_setup_object(telescope, laser_1, separation, analytic_method)
        setup_21 = laser_telescope_setup_object(telescope, laser_2, separation, analytic_method)
        
        gff_11 = setup_11.gff_calculation_unfixed_distance(x, y, z)
        gff_21 = setup_21.gff_calculation_unfixed_distance(x, y, z)
        
        plt.figure(figsize = figsize)
        plt.plot(z, gff_11, label = "$\mathcal{U}$ beam analytically")
        plt.plot(z, gff_21, label = "$\mathcal{N}$ beam analytically")
            
    #plt.plot(z, np.ones(z.shape[0]), '--')
    plt.legend()
    plt.xlabel("$z$ [m]")
    plt.ylabel("GFF")
    plt.show()
    
    t_nb = 1000
    t = np.linspace(0, 2*np.pi, t_nb)
    
    x_telescope = telescope.beam_width(setup_at_z) * np.cos(t)
    y_telescope = telescope.beam_width(setup_at_z) * np.sin(t)
    
    x_laser_uniform = laser_1.beam_center(setup_at_z)[0] + laser_1.beam_width(setup_at_z) * np.cos(t)
    y_laser_uniform = laser_1.beam_center(setup_at_z)[1] + laser_1.beam_width(setup_at_z) * np.sin(t)
    
    x_laser_normal = laser_2.beam_center(setup_at_z)[0] + laser_2.beam_width(setup_at_z) * np.cos(t)
    y_laser_normal = laser_2.beam_center(setup_at_z)[1] + laser_2.beam_width(setup_at_z) * np.sin(t)
    
    plt.figure(figsize = figsize)
    plt.plot(x_telescope, y_telescope, label = "Telescope beam")
    plt.plot(x_laser_uniform, y_laser_uniform, label = "Laser beam following $\mathcal{U}$ law")
    plt.plot(x_laser_normal, y_laser_normal, label = "Laser beam following $\mathcal{N}$ law")
    plt.plot(0, 0, 'x')
    plt.plot(laser_1.beam_center(setup_at_z)[0], laser_1.beam_center(setup_at_z)[1], 'x', color = 'C3')
    plt.plot(laser_1.center[0], laser_1.center[1], 'x', label = "Initial laser center position")
    plt.title("Laser-telescope setup at distance $z =" + str(setup_at_z) + "$ m")
    plt.axis('equal')
    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    if npl.norm(gff_11) != 0:
        z_contact_1 = int(z[np.min(np.argwhere(gff_11 != 0)) - 1])
        z_contact_2 = int(z[np.min(np.argwhere(gff_11 != 0))])
        z_max_1 = int(z[np.argmax(gff_11) - 1])
        z_max_2 = int(z[np.argmax(gff_11)])
        print("Contact telescope-laser at distance z between", z_contact_1, "and", z_contact_2, "m.")
        print("Maximum GFF obtained at distance z between", z_max_1, "and", z_max_2, "m.")
    else:
        print("No contact.")
    
def gff_display():
    interact_manual(interactive_gff_display,
                    Tlscp_radius = "70e-3",
                    Tlscp_angle = "2e-3",
                    Laser_radius = "2e-3",
                    Laser_angle = "2e-3",
                    Separation = "10e-2",
                    Alpha = "0e-3",
                    Beta = "0e-3",
                    Analytic = True,
                    Monte_Carlo = False,
                    Z_min = "1",
                    Z_max = "150",
                    Z_nb = "1000",
                    Setup_at_z = "1")
    
def gff_function_with_uniform_law(z, telescope_radius, telescope_angle, laser_radius, laser_angle, separation, alpha, beta):
    laser_center = np.array([separation, 0])
    focal_length = 1

    telescope = telescope_object(telescope_radius, telescope_angle, focal_length)
    laser = laser_object(laser_radius, laser_angle, laser_center, alpha, beta, "circular_uniform_law")
    
    mesh_nb = 250
    maximum_telescope_radius = telescope.beam_width(z[-1])
    x = np.linspace(-maximum_telescope_radius, maximum_telescope_radius, mesh_nb)
    y = np.linspace(-maximum_telescope_radius, maximum_telescope_radius, mesh_nb)
    
    setup = laser_telescope_setup_object(telescope, laser, separation, analytic_method)
    
    gff = setup.gff_calculation_unfixed_distance(x, y, z)
    
    return gff

def gff_function_with_normal_law(z, telescope_radius, telescope_angle, laser_radius, laser_angle, separation, alpha, beta):
    laser_center = np.array([separation, 0])
    focal_length = 1

    telescope = telescope_object(telescope_radius, telescope_angle, focal_length)
    laser = laser_object(laser_radius, laser_angle, laser_center, alpha, beta, "normal_law")
    
    mesh_nb = 250
    maximum_telescope_radius = telescope.beam_width(z[-1])
    x = np.linspace(-maximum_telescope_radius, maximum_telescope_radius, mesh_nb)
    y = np.linspace(-maximum_telescope_radius, maximum_telescope_radius, mesh_nb)
    
    setup = laser_telescope_setup_object(telescope, laser, separation, analytic_method)
    
    gff = setup.gff_calculation_unfixed_distance(x, y, z)
    
    return gff
    
def optimization_with_uniform_law():
    initial_parameters = np.array([77.5e-3, 1.5e-3, 1.75e-3, 3.5e-3, 10e-2, 4e-3, 0.5e-3])
    bounds_min = np.array([65e-3, 0.5e-3, 1.5e-3, 2e-3, 5e-2, 2e-3, 0e-3])
    bounds_max = np.array([90e-3, 2.5e-3, 2e-3, 5e-3, 15e-2, 6e-3, 1e-3])
    result = opt.curve_fit(gff_function_with_uniform_law, smooth_overlap[0], smooth_overlap[1], initial_parameters, bounds = (bounds_min, bounds_max))
    return result[0]

def optimization_with_normal_law():
    initial_parameters = np.array([77.5e-3, 1.5e-3, 1.75e-3, 3.5e-3, 10e-2, 4e-3, 0.5e-3])
    bounds_min = np.array([65e-3, 0.5e-3, 1.5e-3, 2e-3, 5e-2, 2e-3, 0e-3])
    bounds_max = np.array([90e-3, 2.5e-3, 2e-3, 5e-3, 15e-2, 6e-3, 1e-3])
    result = opt.curve_fit(gff_function_with_normal_law, smooth_overlap[0], smooth_overlap[1], initial_parameters, bounds = (bounds_min, bounds_max))
    return result[0]