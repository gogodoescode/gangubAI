from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'gangubai_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gogodon',
    maintainer_email='gogodon@todo.todo',
    description='GangubAI motor + wander controllers, URDF and Gazebo world',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motor_controller = gangubai_control.motor_controller_node:main',
            'wander_controller = gangubai_control.wander_controller_node:main',
        ],
    },
)
