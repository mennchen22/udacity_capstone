import os
from styx_msgs.msg import TrafficLight
import numpy as np
import rospy
import keras
import cv2


class TLClassifier(object):
    def __init__(self, classifier_model=""):
        
        # Set traffic light state
        self.current_light = TrafficLight.UNKNOWN
        self.model = None
        
         #Get classification model from file
        cwd = os.path.dirname(os.path.realpath(__file__))

        rospy.loginfo("[TLClassifier] Load model from {}".format(classifier_model))

        # Load the PyTorch model
        if classifier_model != "":
            self.model = keras.models.load_model(classifier_model)

        num_classes = 4
        self.classes = {
                0: TrafficLight.UNKNOWN,
                1: TrafficLight.GREEN,
                2: TrafficLight.YELLOW,
                3: TrafficLight.RED
            }

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #   TODO implement light color prediction
        current_light = np.random.choice(
            [TrafficLight.UNKNOWN, TrafficLight.RED,
            TrafficLight.GREEN, TrafficLight.YELLOW]
            )
        # current_light = TrafficLight.GREEN
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (im_width, im_height, _) = image_rgb.shape
        image_np = np.expand_dims(image_rgb, axis=0)/255.0
        rospy.logdebug_throttle(50,"Image detection: ")
        rospy.logdebug_throttle(50,image_np.shape)

        if self.model:
            pass


        ## TODO eval model with image
        rospy.logdebug("Classified traffic light %s", current_light)
        return current_light
