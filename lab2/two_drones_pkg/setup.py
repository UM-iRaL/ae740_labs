from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'two_drones_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files.
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        # Include all files from the 'config' directory
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        # Include all files from the 'mesh' directory
        (os.path.join('share', package_name, 'mesh'), glob('mesh/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='atharva',
    maintainer_email='anavsalkar@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'frames_publisher_node = two_drones_pkg.frames_publisher_node:main',
            'plots_publisher_node = two_drones_pkg.plots_publisher_node:main',
        ],
    },
)
