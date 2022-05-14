import time

import database_manager as dbm
import detect_features as df
import cv2


def main(demo_mode=False):
    old_features = None
    current_people = 0
    cam = cv2.VideoCapture(0)
    running = True
    while running:
        result, image = cam.read()
        if result:
            face_count = len(df.get_faces(image))
            if current_people == 0:
                #
                time.sleep(.2)
                #result, image = cam.read()
                if face_count > 0:
                    current_people = 1
                    features = df.get_features(image)
                    if old_features is None or old_features != features:
                        dbm.add_features_to_database(features)
                        old_features = features
                        print("adding to db")
                        if demo_mode:
                            x = input("enter q to stop")
                            if x == "q":
                                running = False
            else:
                if face_count == 0:
                    current_people = 0
        time.sleep(.1)
    cam.release()


if __name__ == "__main__":
    main(True)
