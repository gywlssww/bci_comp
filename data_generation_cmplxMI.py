import numpy as np
import pandas as pd
from filtering import arr_bandpass_filter
class complexMIDataset:
    def __init__(self, eventdata='./mi_subject11_event.csv',sigdata='./mi_subject11_data.csv'):
        
        self.data = pd.read_csv(sigdata)
        self.eventdata=pd.read_csv(eventdata)
        self.Fs = 250  # 250Hz from original paper

        # keys of data ['s', 'etyp', 'epos', 'edur', 'artifacts']

        self.events_type = self.eventdata['type'].tolist()
        self.events_position = self.eventdata['position'].tolist()
        self.events_duration = self.eventdata['duration'].tolist()
        #self.artifacts = self.eventdata['artifacts']

        # Types of motor imagery
        self.mi_types = {1536: 'elbow flexion', 1537: 'elbow extension',
                         1538: 'supination', 1539: 'pronation', 1540: 'hand close',1541:'hand open',1542:'rest'}

    def get_trials_from_channel(self, channel=7):
        startrial_code = 768
        starttrial_events = self.events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]
        trials = []
        classes = []
        for index in idxs:
            try:
                type_e = self.events_type[0, index+1]
                class_e = self.mi_types[type_e]
                classes.append(class_e)

                start = self.events_position[0, index]
                stop = start + self.events_duration[0, index]
                trial = self.raw[channel, start:stop]
                trial = trial.reshape((1, -1))
                trials.append(trial)

            except:
                continue

        return trials, classes

    def get_trials_from_channels(self, channels=[7, 9, 11]):
        trials_c = []
        classes_c = []
        for c in channels:
            t, c = self.get_trials_from_channel(channel=c)

            tt = np.concatenate(t, axis=0)
            trials_c.append(tt)
            classes_c.append(c)

        return trials_c, classes_c


def load_cnt_mrk_y(sigpath,eventpath):
    ds = complexMIDataset(eventdata=eventpath,sigdata=sigpath)
    cnt = ds.data
    mrk = ds.events_position
    dur = ds.events_duration
    y = np.zeros(len(mrk))
    for i in range(len(mrk)):
        
        event_type=ds.events_type[i]
        print(event_type)
        if event_type not in ds.mi_types.keys():
            y[i] = 8
        elif ds.mi_types[event_type] == 'elbow flexion':
            y[i] = 0
        elif ds.mi_types[event_type] == 'elbow extension':
            y[i] = 1
        elif ds.mi_types[event_type] == 'supination':
            y[i] = 2
        elif ds.mi_types[event_type] == 'pronation':
            y[i] = 3
        elif ds.mi_types[event_type] == 'hand close':
            y[i] = 4
        elif ds.mi_types[event_type] == 'hand open':
            y[i] = 5
        elif ds.mi_types[event_type] == 'rest':
            y[i] = 6
        else:
            y[i] = 7
    #return cnt, mrk[:-1], dur[:-1], y[1:]
    return cnt, mrk, dur, y
   

def cnt_to_epo(cnt, mrk, dur):
    epo = []
    for i in range(len(mrk)):
        epo.append(np.array(cnt[mrk[i] : mrk[i] + dur[i]]))
    return np.array(epo)

def out_label_remover(x, y):
    new_x = []
    new_y = []
    for i in range(len(y)):
        if y[i] != 7 and y[i] != 8:
            new_x.append(np.array(x[i]))
            new_y.append(int(y[i]))
    return np.array(new_x), np.array(new_y)


def gen_filtered_data():
    data = []
    
    eventpath = './mi_subject11_event.csv'
    sigpath='./mi_subject11_data.csv'
    cnt, mrk, dur, y = load_cnt_mrk_y(sigpath,eventpath)
    f_cnt = arr_bandpass_filter(cnt, 8, 30, 250, 5)
    epo = cnt_to_epo(f_cnt, mrk, dur)
    epo, y = out_label_remover(epo, y)
    new_epo = []
    for v in epo:
        new_epo.append(v[750:1500,:])
    new_epo = np.array(new_epo)
    data.append({'x': new_epo, 'y': np.squeeze(y.T)})
    np.savez_compressed('data/test.npz', data = data)
        # scipy.io.savemat('data/c_iv_2a_epo/' + str(i) + '.mat', {'x': new_epo, 'y': np.squeeze(y.T)})




gen_filtered_data()