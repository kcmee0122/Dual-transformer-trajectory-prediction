import numpy as np

def Draw_vehicle_pts(Track_data):

    x_data  = Track_data['xCenter']
    y_data  = Track_data['yCenter']
    width   = Track_data['width']
    length  = Track_data['length']
    heading = np.deg2rad(Track_data['heading'])
    
    c1 = [x_data + length/2*np.cos(heading) + width/2*np.sin(heading), y_data + length/2*np.sin(heading) - width/2*np.cos(heading)]
    c2 = [x_data + length/2*np.cos(heading) - width/2*np.sin(heading), y_data + length/2*np.sin(heading) + width/2*np.cos(heading)]
    c3 = [x_data - length/2*np.cos(heading) - width/2*np.sin(heading), y_data - length/2*np.sin(heading) + width/2*np.cos(heading)]
    c4 = [x_data - length/2*np.cos(heading) + width/2*np.sin(heading), y_data - length/2*np.sin(heading) - width/2*np.cos(heading)]

    Trans = np.array([c1, c2, c3, c4])

    return Trans

