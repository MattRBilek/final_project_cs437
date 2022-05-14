import os

import detect_features as df
import matplotlib.pyplot as plt
from cred import firebaseConfig
import datetime
from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("config.json")
firebase_admin.initialize_app(cred)

credential_path = u'config.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

db = firestore.Client(project=u'image-detection-ashwin-matt')


def filter_today_only(features):
    return features['time'].replace(tzinfo=None) >= datetime.datetime.now() - datetime.timedelta(days=1)


def add_features_to_database(features):
    current_time = datetime.datetime.now()
    features['time'] = current_time
    db.collection(u'people').add(features)


def get_last_person_to_walk_in():
    snapshot = db.collection('people').order_by("time", direction=firestore.Query.DESCENDING).limit(1)
    results = snapshot.get()

    for doc in results:
        return doc.to_dict()


def visualize_hair(filter=None):
    snapshot = db.collection('people').stream()
    counts = {}
    for result in snapshot:
        color = result.to_dict()['hair']
        if filter is None or filter(result.to_dict()):
            counts[color] = counts.get(color, 0) + 1

    names = list(counts.keys())
    values = list(counts.values())

    plt.bar(range(len(counts)), values, tick_label=names)
    plt.show()


def visualize_shirt(filter=None):
    snapshot = db.collection('people').stream()
    counts = {}
    for result in snapshot:
        color = result.to_dict()['shirt']
        if filter is None or filter(result.to_dict()):
            counts[color] = counts.get(color, 0) + 1

    names = list(counts.keys())
    values = list(counts.values())

    plt.bar(range(len(counts)), values, tick_label=names)
    plt.show()


if __name__ == "__main__":
    # add_features_to_database(["blond","white",47])
    print(get_last_person_to_walk_in())
    visualize_hair()
    visualize_shirt(filter_today_only)
