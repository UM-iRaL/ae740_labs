from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'controller_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Add launch files [LAB 4]
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='atharva',
    maintainer_email='anavsalkar@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'crazyflie_mpc_controller = controller_pkg.crazyflie_mpc:main',
            'target_tracking_controller = controller_pkg.target_tracking_mpc:main', # [LAB 4]
        ],
    },
)