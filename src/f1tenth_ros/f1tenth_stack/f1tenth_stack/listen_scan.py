#!/usr/bin/env python

# Author: christoph.roesmann@tu-dortmund.de
# Modifier: Donghee Han, hdh7485@kaist.ac.kr

from multiprocessing import get_logger
from pkgutil import get_loader
import rclpy
from rclpy.node import Node
import math

from sensor_msgs.msg import LaserScan
from collections import deque
import numpy as np

class ScanListener(Node):
  def __init__(self):
    super().__init__('scan_listener')

    self.sub = self.create_subscription(
      LaserScan,
      'scan',
      self.scanCallback,
      1
    )

  def scanCallback(self, msg):
    a = list(msg.ranges)
    self.get_logger().info('%d' % len(a))

def main(args=None):
    rclpy.init(args=args)
    p = ScanListener()
    rclpy.spin(p)

if __name__ == '__main__':
    main()