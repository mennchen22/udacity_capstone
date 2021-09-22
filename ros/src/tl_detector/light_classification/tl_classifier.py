import os
from styx_msgs.msg import TrafficLight
import numpy as np
import rospy
import keras
import cv2


class TLClassifier(object):
    def __init__(self, classifier_model="tl_classifier_leenet5.h5"):

        # Set traffic light state
        self.current_light = TrafficLight.UNKNOWN
        self.model = None

        
        # Load the PyTorch model
        if classifier_model != "":
             # Get classification model from file
            cwd = os.path.dirname(os.path.realpath(__file__))
            classifier_model = cwd + os.sep + classifier_model
            rospy.loginfo(
            "[TLClassifier] Load model from {}".format(classifier_model))

            self.model = keras.models.load_model(classifier_model)
        else:
             rospy.loginfo("[TLClassifier] No model in use, take random traffic light output")

        self.classes = {
            0: TrafficLight.RED,
            1: TrafficLight.GREEN,
            2: TrafficLight.UNKNOWN,
            3: TrafficLight.YELLOW
        }

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #   TODO implement light color prediction

        current_light = TrafficLight.UNKNOWN
        if self.model and image is not None:
            # Preprocessing
            image_p = self.preprocess_image(image)
            #rospy.loginfo(200, "Image detection: ")
            #rospy.loginfo(200, image_p.shape)
            current_light_inx = np.argmax(self.model.predict(image_p))
            current_light = self.classes[current_light_inx]
        else:
            current_light = np.random.choice(
                [TrafficLight.UNKNOWN,
                 TrafficLight.RED,
                 TrafficLight.GREEN,
                 TrafficLight.YELLOW
                 ])

        # TODO eval model with image
        rospy.logdebug_throttle(10,"Classified traffic light %s", current_light)
        return current_light

    @staticmethod
    def preprocess_image(image, img_size=(180, 180)):
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = cv2.resize(image_np, img_size)
        image_np = np.expand_dims(image_np, axis=0)/255.0
        image_np = np.asarray(image_np) - 0.5
        return image_np
