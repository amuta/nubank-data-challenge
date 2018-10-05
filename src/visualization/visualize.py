import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls


class PlotBoxR(object):
    
    
    def Trace(self,feat_names,value): 
    
        trace = go.Box(
            y=value,
            name = feat_names,
            marker = dict(
                color = 'rgb(0, 128, 128)',
            )
        )
        return trace

    def PlotResult(self,names,results):
        
        data = []

        for i in range(len(names)):
            data.append(self.__Trace(names[i],results[i]))

        py.iplot(data)