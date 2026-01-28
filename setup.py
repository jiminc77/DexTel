from setuptools import setup

package_name = 'dextel'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@todo.todo',
    description='DexTel Hand Tracking Package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ur3_realsense = dextel.ur3_realsense:main',
            'ur3_vision = dextel.ur3_vision:main',
            'ur3_realsense_hamer = dextel.ur3_realsense_hamer:main',
        ],
    },
)
