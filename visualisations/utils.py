import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from collections import Counter
import plotly.graph_objects as go

CATS = ['ADJ_Gender', 'ADJ_Number',  'NOUN_Number', 'NOUN_Gender', 'NOUN_Case', 'VERB_Aspect', 'VERB_Tense']

bert_neurons_layers = []
i = 0
d = np.arange(0,9984).tolist()
while True:
    bert_neurons_layers.append(d[i:i+768])
    i = i+768
    if i == len(d):
        break

def accuracy_lines(dct1, dct2, dct3, dct4, dct5, dct6, cat):
    
    l1 = [v[1]['__OVERALL__'] for k, v in dct1.items()]
    l2 = [v[1]['__OVERALL__'] for k, v in dct2.items()]
    l3 = [v[1]['__OVERALL__'] for k, v in dct3.items()]
    accuracy_test1 = [round((i + j + r)/3, 2) for i, j, r in zip(l1, l2, l3)]
    
    l4 = [v[1]['__OVERALL__'] for k, v in dct4.items()]
    l5 = [v[1]['__OVERALL__'] for k, v in dct5.items()]
    l6 = [v[1]['__OVERALL__'] for k, v in dct6.items()]
    accuracy_test2 = [round((i + j + r)/3, 2) for i, j, r in zip(l4, l5, l6)]
    
    l = [k for k in dct1.keys()]
    d = pd.DataFrame({'Layers': l, 'good model' : accuracy_test1, 'broken model': accuracy_test2})    
    fig = px.line(d, x='Layers', y=['good model', 'broken model'], template="plotly_white",
                 color_discrete_map = {'good model': 'green', 'broken model': 'red'})
    
    fig.add_trace(go.Scatter(x=l, y=l1, name='good 1',
                         line = dict(color='green', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=l, y=l2, name='good 2',
                             line=dict(color='green', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=l, y=l3, name='good 3',
                             line=dict(color='green', width=1, dash='dot')))
    
    fig.add_trace(go.Scatter(x=l, y=l4, name='broken 1',
                         line = dict(color='red', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=l, y=l5, name='broken 2',
                             line=dict(color='red', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=l, y=l6, name='broken 3',
                             line=dict(color='red', width=1, dash='dot')))
    
    fig.update_xaxes(tickmode='linear')
    fig.update_yaxes(title='Accuracy')
    #fig.update_layout(title_text=f"{cat} test accuracy: model comparison", title_x=0.4)
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1,
    xanchor="right",
    x=1, traceorder="normal",font=dict(size=14)))
    fig.update_layout(
    font=dict(
        family="Times New Roman",
        size=14,  # Set the font size here
        color="Black"))
    fig.show()

def accuracy_dif(d1, d2, d3, d4, d5, d6):
    
    cats=CATS
    dct_acc1 = {}
    dct_acc2 = {}
    dct_acc3 = {}
    dct_acc4 = {}
    dct_acc5 = {}
    dct_acc6 = {}
    
    for c in cats:
        dct_acc1[c] = d1[c]
        dct_acc2[c]  = d2[c]  
        dct_acc3[c]  = d3[c] 
        dct_acc4[c]  = d4[c] 
        dct_acc5[c]  = d5[c] 
        dct_acc6[c]  = d6[c] 
        
        
    accuracy_test1 = [v[1]['__OVERALL__'] for k, v in dct_acc1.items()]
    accuracy_test2 = [v[1]['__OVERALL__'] for k, v in dct_acc2.items()]
    accuracy_test3 = [v[1]['__OVERALL__'] for k, v in dct_acc3.items()]
    accuracy_test11 = [round((i + j + r)/3, 2) for i, j, r in zip(accuracy_test1, accuracy_test2, accuracy_test3)]
    
    
    accuracy_test4 = [v[1]['__OVERALL__'] for k, v in dct_acc4.items()]
    accuracy_test5 = [v[1]['__OVERALL__'] for k, v in dct_acc5.items()]
    accuracy_test6 = [v[1]['__OVERALL__'] for k, v in dct_acc6.items()]
    accuracy_test21 = [round((i + j + r)/3, 3) for i, j, r in zip(accuracy_test4, accuracy_test5, accuracy_test6)]
    
    d = pd.DataFrame({'Categories': cats, 'good model' : accuracy_test11, 'broken model': accuracy_test21})    
    fig = px.bar(d, x='Categories', y=['good model', 'broken model'], template="plotly_white", barmode='group',
                 color_discrete_map = {'good model': 'seagreen', 'broken model': 'coral'})
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    fig.update_yaxes(title='Accuracy')
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1,
    xanchor="right",
    x=1, traceorder="normal",font=dict(size=14)
    ))
    fig.update_layout(
    font=dict(
        family="Times New Roman",
        size=14,  # Set the font size here
        color="Black"
    ))
    fig.show()

def accuracy_dif_control(d1, d2, d3, d4, d5, d6):
    
    cats=CATS
    dct_acc1 = {}
    dct_acc2 = {}
    dct_acc3 = {}
    dct_acc4 = {}
    dct_acc5 = {}
    dct_acc6 = {}
    
    for c in cats:
        dct_acc1[c] = d1[c]
        dct_acc2[c]  = d2[c]  
        dct_acc3[c]  = d3[c] 
        dct_acc4[c]  = d4[c] 
        dct_acc5[c]  = d5[c] 
        dct_acc6[c]  = d6[c] 
        
        
    accuracy_test1 = [v[1]['__OVERALL__'] for k, v in dct_acc1.items()]
    accuracy_test2 = [v[1]['__OVERALL__'] for k, v in dct_acc2.items()]
    accuracy_test3 = [v[1]['__OVERALL__'] for k, v in dct_acc3.items()]
    accuracy_test11 = [round((i + j + r)/3, 2) for i, j, r in zip(accuracy_test1, accuracy_test2, accuracy_test3)]
    
    
    accuracy_test4 = [v[1]['__OVERALL__'] for k, v in dct_acc4.items()]
    accuracy_test5 = [v[1]['__OVERALL__'] for k, v in dct_acc5.items()]
    accuracy_test6 = [v[1]['__OVERALL__'] for k, v in dct_acc6.items()]
    accuracy_test21 = [round((i + j + r)/3, 2) for i, j, r in zip(accuracy_test4, accuracy_test5, accuracy_test6)]
    
    d = pd.DataFrame({'Categories': CATS, 'actual accuracy' : accuracy_test11, 'control task': accuracy_test21})    
    fig = px.bar(d, x='Categories', y=['actual accuracy', 'control task'], template="plotly_white", barmode='group', 
                color_discrete_map = {'actual accuracy': 'green', 'control task': 'red'})
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    fig.update_xaxes(tickmode='linear')
    fig.update_yaxes(title='Accuracy')
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1,
    xanchor="right",
    x=1, traceorder="normal",font=dict(size=14)
    ))
    fig.update_layout(
    font=dict(
        family="Times New Roman",
        size=14,  # Set the font size here
        color="Black"
    ))
    fig.show()

def accuracy_dif2(d1, d2, d3, d4, d5, d6, d7, d8, d9):
    
    cats=CATS
    dct_acc1 = {}
    dct_acc2 = {}
    dct_acc3 = {}
    dct_acc4 = {}
    dct_acc5 = {}
    dct_acc6 = {}
    dct_acc7 = {}
    dct_acc8 = {}
    dct_acc9 = {}
    
    
    for c in cats:
        dct_acc1[c] = d1[c]
        dct_acc2[c]  = d2[c]  
        dct_acc3[c]  = d3[c] 
        dct_acc4[c]  = d4[c] 
        dct_acc5[c]  = d5[c] 
        dct_acc6[c]  = d6[c] 
        dct_acc7[c]  = d7[c] 
        dct_acc8[c]  = d8[c] 
        dct_acc9[c]  = d9[c] 
        
    accuracy_test1 = [v[1]['__OVERALL__'] for k, v in dct_acc1.items()]
    accuracy_test2 = [v[1]['__OVERALL__'] for k, v in dct_acc2.items()]
    accuracy_test3 = [v[1]['__OVERALL__'] for k, v in dct_acc3.items()]
    accuracy_test11 = [round((i + j + r)/3, 2) for i, j, r in zip(accuracy_test1, accuracy_test2, accuracy_test3)]
    
    
    accuracy_test4 = [v[1]['__OVERALL__'] for k, v in dct_acc4.items()]
    accuracy_test5 = [v[1]['__OVERALL__'] for k, v in dct_acc5.items()]
    accuracy_test6 = [v[1]['__OVERALL__'] for k, v in dct_acc6.items()]
    accuracy_test21 = [round((i + j + r)/3, 2) for i, j, r in zip(accuracy_test4, accuracy_test5, accuracy_test6)]
    
    accuracy_test7 = [v[1]['__OVERALL__'] for k, v in dct_acc7.items()]
    accuracy_test8 = [v[1]['__OVERALL__'] for k, v in dct_acc8.items()]
    accuracy_test9 = [v[1]['__OVERALL__'] for k, v in dct_acc9.items()]
    accuracy_test31 = [round((i + j + r)/3, 2) for i, j, r in zip(accuracy_test7, accuracy_test8, accuracy_test9)]
    
    
    d = pd.DataFrame({'Categories': cats, 'all neurons' : accuracy_test11, 'top 20%': accuracy_test21, 
                      'bottom 20%': accuracy_test31})    
    fig = px.bar(d, x='Categories', y=['all neurons', 'top 20%', 'bottom 20%'], template="plotly_white", barmode='group') 
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    fig.update_yaxes(title='Accuracy')
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1,
    xanchor="right",
    x=1, traceorder="normal",font=dict(size=14)))
    fig.update_layout(
    font=dict(
        family="Times New Roman",
        size=14,  # Set the font size here
        color="Black"))
    fig.show()

def plot_distr(dct1, dct2, dct3):
    d1 = {}
    d2 = {}
    d3 = {}
    for c in CATS:
        d1[c] = len(dct1[c])
        d2[c] = len(dct2[c])
        d3[c] = len(dct3[c])
        
    accuracy_test1 = [v for k, v in d1.items()]
    accuracy_test2 = [v for k, v in d2.items()]
    accuracy_test3 = [v for k, v in d3.items()]
    accuracy_test11 = [round((i + j + r)/3) for i, j, r in zip(accuracy_test1, accuracy_test2, accuracy_test3)]
    
        
    def valuelabel(cc):
        for i in range(7):
            plt.text(i,cc[i],cc[i], ha = 'center',
                     bbox = dict(facecolor = 'cyan', alpha =0.7), size='small')
                
    fig = plt.figure(figsize=(5,5))
    col_map = plt.get_cmap('Paired')
    plt.xlabel('Categories', fontsize=8) 
    plt.ylabel('Number of top-20% of neurons', fontsize=8)
    plt.bar(list(d1.keys()), accuracy_test11, 
            color=col_map.colors, edgecolor='k', width=0.5)
    valuelabel([v for v in d1.values()])
    plt.xticks(rotation=30, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig('foo5.png')
    plt.show()


def loss(path):
    with open(path) as file:
        lines = [line.rstrip() for line in file]
    all_l = []
    all_s = []
    for l in lines:
        all_l.append(float(l.split()[1]))
        all_s.append(l.split()[2][:-3]+'k')
    loss = []
    steps = []
    loss.append(float(lines[1].split()[1]))
    steps.append(lines[1].split()[2][:-3]+'k')
    for i in range(50, 1050, 50):
        loss.append(float(lines[i].split()[1]))
        steps.append(lines[i].split()[2][:-3]+'k')
    return loss, steps


def common_neurons_percentage_multiple(dct_acc1, dct_acc2, le=[10, 20, 30]):
    cats = CATS
    d1 = {}
    d2 = {}
    for c in cats:
        d1[c] = dct_acc1[c]
        d2[c]  = dct_acc2[c] 
    common_cats = []
    for k1 in d1.keys():
        for k2 in d2.keys():   
            if k1 == k2:
                common_cats.append(k1)
    le = sorted(le, reverse=True)
    le_id = [str(i) + '%' for i in le]        
    df = pd.DataFrame(index=le_id, columns=common_cats)
    df = df.fillna(0)

    for cat in common_cats:
        common_neurons = []
        for l in le:
            p = len(set(d1[cat][0][:d1[cat][1][l]]) & set(d2[cat][0][:d2[cat][1][l]])) * 100 / len((set(d1[cat][0][:d1[cat][1][l]]) | set(d2[cat][0][:d2[cat][1][l]])))
            common_neurons.append(round(p, 2))
        df[cat] = common_neurons
    return df 


def counter_layers(dct):
    k = []
    for i in dct:
        for j in bert_neurons_layers:
            for m in j:
                if i==m:
                    k.append(bert_neurons_layers.index(j))
    keys = Counter(k).keys()
    values = Counter(k).values()
    value = [str(round(v / sum(values) *100,1))+'%' for v in values]
    neurons = dict(zip(keys, values))
    new = dict(zip(keys, value))
    return dict(sorted(new.items())), dict(sorted(neurons.items()))

def mine(d1, d2, d3, idx):
    cats = CATS
    df = pd.DataFrame(index = cats, columns=[0,1,2,3,4,5,6,7,8,9,10,11,12])
    df = df.fillna(0)
    for cat in cats:
        new_good, new_neurons1 = counter_layers(d1[cat.split()[0]])
        new_good, new_neurons2 = counter_layers(d2[cat.split()[0]])
        new_good, new_neurons3 = counter_layers(d3[cat.split()[0]])
        
        accuracy_test1 = [v for k, v in new_neurons1.items()]
        accuracy_test2 = [v for k, v in new_neurons2.items()]
        accuracy_test3 = [v for k, v in new_neurons3.items()]
        accuracy_test11 = [round((i + j + r)/3, 2) for i, j, r in zip(accuracy_test1, accuracy_test2, accuracy_test3)]
        new = dict(zip(new_neurons1.keys(), accuracy_test11))
        df.loc[cat] = pd.Series(new)
    return df

def large_vi(df1, df2, n, k):
    
    layers = list(df1.columns)
    layers = [int(l) for l in layers]
    cats = df1.index
    cats = [cat.split()[0] for cat in cats]
    
    index1 = df1.index
    index2 = df2.index

    a = 2  # number of rows
    b = 3  # number of columns
    c = 1  # initialize plot counter

    fig, ax = plt.subplots(2, 3,figsize=(14,9))#, gridspec_kw={'width_ratios': [0.33,0.33,0.33]})
    col_map = plt.get_cmap('Paired')
    fig.suptitle(f'Top 20% neurons per category: layer-wise distribution for probed BERT models on {k} steps', fontsize=14)
    i = 0
    while True:
        try:
            plt.subplot(a, b, c)
            plt.title(f'{index1[i].split()[0]}')
            plt.ylabel('Number of neurons per layer') 
            plt.xlabel('Layers') 
            plt.plot(layers, df1.loc[index1[i]], label = "good model", color="g")
            plt.plot(layers, df2.loc[index2[i]], label = "broken model", color="r")
            plt.xticks(range(0,len(layers)),layers)
            plt.legend(loc='best')
            plt.grid()
            c = c + 1
            i+=1
            if i ==6:
                break
        except IndexError:
            break
    plt.tight_layout()
    plt.savefig(f'foo{n}.png')
    plt.show()