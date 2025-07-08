import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

df = pd.read_csv(os.path.join(os.getcwd(),r'acc_maps.csv'))

tmp = df.columns.to_list()
print(len(df.columns.to_list()))


#find all static data and remove

static_data = []

for item in tmp:
    if item.startswith('Static'):
        static_data.append(item)

df = df.drop(columns=static_data)


#remove graphics car coordinates since it is a vector contains x,y,z and not well formated.
test = df["Graphics_car_coordinates"]
df = df.drop(["Graphics_car_coordinates"],axis=1)

xs = []
ys = []
zs = []

# split the vec<x,y,z> in graphics_car_coordinates into x,y,z

for item in test:
    #tmp = test.iloc[0]
    vec = item[item.find('(')+1:item.find(')')]
    x = vec[vec.find('x')+2:vec.find(',')]
    y = vec[vec.find('y')+2:vec.find('z')-2]
    z = vec[vec.find('z')+2:]
    xs.append(x)
    ys.append(y)
    zs.append(z)
# add the data into the dataframce
df["Graphics_car_coordinate_x"] = xs
df["Graphics_car_coordinate_y"] = ys
df["Graphics_car_coordinate_z"] = zs


# remove invalid lap

df = df[df['Graphics_is_valid_lap'] == True]

feature_list = ["Physics_wheel_angular_s_front_left",
                "Physics_wheel_angular_s_front_right",
                "Physics_wheel_angular_s_rear_left",
                "Physics_wheel_angular_s_rear_right",
                "Physics_slip_ratio_front_left",
                "Physics_slip_ratio_front_right",
                "Physics_slip_ratio_rear_left",
                "Physics_slip_ratio_rear_right",
                "Physics_slip_angle_front_left",
                "Physics_slip_angle_front_right",
                "Physics_slip_angle_rear_left",
                "Physics_slip_angle_rear_right",
                "Physics_wheel_slip_front_left",
                "Physics_wheel_slip_front_right",
                "Physics_wheel_slip_rear_left",
                "Physics_wheel_slip_rear_right",
                "Physics_brake_pressure_front_left",
                "Physics_brake_pressure_front_right",
                "Physics_brake_pressure_rear_left",
                "Physics_brake_pressure_rear_right",
                "Physics_brake_temp_front_left",
                "Physics_brake_temp_front_right",
                "Physics_brake_temp_rear_left",
                "Physics_brake_temp_rear_right",
                "Physics_tyre_contact_normal_front_left_x",
                "Physics_tyre_contact_normal_front_left_y",
                "Physics_tyre_contact_normal_front_left_z",
                "Physics_tyre_contact_normal_front_right_x",
                "Physics_tyre_contact_normal_front_right_y",
                "Physics_tyre_contact_normal_front_right_z",
                "Physics_tyre_contact_normal_rear_left_x",
                "Physics_tyre_contact_normal_rear_left_y",
                "Physics_tyre_contact_normal_rear_left_z",
                "Physics_tyre_contact_normal_rear_right_x",
                "Physics_tyre_contact_normal_rear_right_y",
                "Physics_tyre_contact_normal_rear_right_z",
                "Physics_tyre_contact_point_front_left_x",
                "Physics_tyre_contact_point_front_left_y",
                "Physics_tyre_contact_point_front_left_z",
                "Physics_tyre_contact_point_front_right_x",
                "Physics_tyre_contact_point_front_right_y",
                "Physics_tyre_contact_point_front_right_z",
                "Physics_tyre_contact_point_rear_left_x",
                "Physics_tyre_contact_point_rear_left_y",
                "Physics_tyre_contact_point_rear_left_z",
                "Physics_tyre_contact_point_rear_right_x",
                "Physics_tyre_contact_point_rear_right_y",
                "Physics_tyre_contact_point_rear_right_z",
                "Physics_tyre_core_temp_front_left",
                "Physics_tyre_core_temp_front_right",
                "Physics_tyre_core_temp_rear_left",
                "Physics_tyre_core_temp_rear_right",
                "Graphics_mfd_tyre_pressure_front_left",
                "Graphics_mfd_tyre_pressure_front_right",
                "Graphics_mfd_tyre_pressure_rear_left",
                "Graphics_mfd_tyre_pressure_rear_right",
                "Physics_steer_angle",
                "Physics_gear",
                "Physics_tc",
                "Physics_abs",
                "Graphics_tc_cut_level",
                "Physics_brake_bias",
                "Physics_road_temp",
                "Physics_air_temp",
                "Graphics_wind_speed",
                "Physics_speed_kmh",
                "Physics_brake",
                "Physics_gas"
                ]

df = df[feature_list]

df.to_csv('output_clean_split.csv',index=False)

