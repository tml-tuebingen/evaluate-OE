from sklearn import preprocessing
import numpy as np

def vectorizer_kdd(data, labels):
    # encode labels
    labels = [x.decode("utf-8") for x in labels]

    # separate other categorical labels
    duration = data[:, 0]
    protocol_type = data[:, 1]
    service = data[:, 2]
    flag = data[:, 3]
    rest = data[:, 4]

    # clean up the decoding things
    protocol_type = [x.decode("utf-8") for x in protocol_type]
    service = [x.decode("utf-8") for x in service]
    flag = [x.decode("utf-8") for x in flag]

    # convert string features/labels into integers
    protcoler = preprocessing.LabelEncoder()
    protocol_labels = protcoler.fit(protocol_type).transform(protocol_type)

    servicer = preprocessing.LabelEncoder()
    service_labels = servicer.fit(service).transform(service)

    flagger = preprocessing.LabelEncoder()
    flag_labels = flagger.fit(flag).transform(flag)

    labeller = preprocessing.LabelEncoder()
    integer_labels = labeller.fit(labels).transform(labels)

    vectors = np.zeros((len(data[0]), len(labels)), dtype=float)
    vectors[0, :] = duration[:]
    vectors[1, :] = protocol_labels[:]
    vectors[2, :] = service_labels[:]
    vectors[3, :] = flag_labels[:]
    vectors[4:, :] = np.float32(np.asarray(rest))

    return np.transpose(vectors), np.float32(integer_labels)