import numpy as np
#import matplotlib as mp
#import matplotlib.gridspec as gridspec
#import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import lmfit

#%%
def linear_fit(x=np.array([0]),y=np.array([0]),y_error=np.array([0]),scale=0,show_sigenv=0):
    if np.sum(np.abs(x)+np.abs(y))==0:
        #zum testen
        x=np.arange(100,200,2.5)
        #x=x-np.mean(x)
        y_error=1*np.ones(np.size(x)) # stddev
        #y_error=10*np.arange(np.size(x))+0.01
        offset=0
        slope=0.05
        y=offset+np.random.normal(0, y_error, size=x.size)+x*slope   
    
        
    model=lmfit.models.LinearModel()
    params=model.guess(y,x=x)
    
    if y_error.size==1:
        y_error=y_error*np.ones(np.size(x))
     
    if np.sum(y_error)==0:
        weights_in=np.ones(np.size(x))
        scale_with_red_chi_sq=scale
    else:
        weights_in=1/y_error
        scale_with_red_chi_sq=scale
        print(scale)
    
    out=model.fit(y,params,x=x,weights=weights_in,method='leastsq',scale_covar=scale_with_red_chi_sq)
    uncert=out.eval_uncertainty(sigma=1)
    uncert2=out.eval_uncertainty(sigma=2)
    uncert3=out.eval_uncertainty(sigma=3)
    
    fig=plt.figure(figsize=(6,4),dpi=200)
    print(out.fit_report(min_correl=0.01,correl_mode ='list' ))
    try:
        out.conf_interval()
        print(out.ci_report())
    except:
        print('no conf intervall calculated')
    if show_sigenv:    
        plt.fill_between(x,out.best_fit-uncert3,out.best_fit+uncert3,color='grey',alpha=0.25)
        plt.fill_between(x,out.best_fit-uncert2,out.best_fit+uncert2,color='grey',alpha=0.25)
        plt.fill_between(x,out.best_fit-uncert,out.best_fit+uncert,color='grey')
    
    if np.sum(y_error)==0:
        plt.plot(x,y,'o')
    else:
        if y_error.size==1:
           y_error=y_error*np.ones(np.size(x)) 
        plt.errorbar(x,y,y_error,fmt='o',color='k',capsize=4,capthick=2)
    plt.plot(x,out.best_fit,color='b')
    plt.xlabel('x')
    plt.ylabel('y')
    out.x=x
    out.y=y
    out.y_error=y_error
    out.weights=weights_in
    
    return out,fig


#%% 

def PosErsSigZif(x): 
    '''

    Gibt die Position der ersten signifikanten Ziffer von x an

    '''
    return -int(np.floor(np.log10(abs(x))))


def rundung(wert,fehler=None,zwischen=True):
    '''

    Fehler von Endergebnissen werden mit einer signifikanten Stelle angegeben. Fehler werden mit einer signifikanten
    Stelle aufgerundet. Ergebniswert und Fehler müssen in der gleichen Zehnerpotenz enden. 
    Hierbei wird der Ergebniswert kaufmännisch gerundet.
    Zwischenergebnisse werden mit zwei signifikanten Stellen im Fehler angegeben. In diesem Fall erfolgt die Rundung
    auch in dem Fehler kaufmännisch

    '''

    #Kein Fehler bekannt: Runden auf 1, bzw. 2 Nachkommastellen (bei Zwischenergebnis)
    if fehler is None:
        if zwischen:
            if abs(wert) < 0.1: #wenn Wert kleiner als eine Dezimal-Stellen, wieder runden auf zwei signifikante Stellen
                power = PosErsSigZif(wert) + 1
            else:
                power = 2
        else:
            if abs(wert) < 1: #wenn Wert kleiner als eins, runden auf eine signifikante Stelle
                power = PosErsSigZif(wert)
            else:
                power = 1

        return round(wert,power)
        
    #Fehler bekannt: Runden auf eine, bzw. signifikante Stellen des Fehlers (bei Zwischenergebnis) 
    else:
        if zwischen:
            n=2
        else:
            n=1

        power = PosErsSigZif(fehler) + (n - 1)
        
        if zwischen:
            fehler_round = round(fehler, power)
        else:
            factor = (10 ** power)
            fehler_round = np.ceil(fehler * factor) / factor # hier aufrunden
            
        return round(wert,power), fehler_round



def support(xwerte,ywerte,yfehler):              #Funktionsdefinition
    """Berechnet die Summen über x, x^2, y, xy und 1, jeweils durch den Fehler^2."""
    
    x2=0                                         #Summe über (x^2)/(sigma^2)
    for i in range(len(xwerte)):                 #gehe über alle Werte
        x2 += (xwerte[i]**2) / (yfehler[i]**2)   #und addiere die einzelnen Summenteile auf
    x=0                                          #Summe über (x)/(sigma^2)
    for i in range(len(xwerte)):
        x += xwerte[i] / (yfehler[i]**2)
    y=0                                          #Summe über (y)/(sigma^2)
    for i in range(len(ywerte)):
        y += ywerte[i] / (yfehler[i]**2)
    xy=0                                         #Summe über (x*y)/(sigma^2)
    for i in range(len(xwerte)):
        xy += (xwerte[i]*ywerte[i]) / (yfehler[i]**2)
    eins=0                                       #Summe über (1)/(sigma^2)
    for i in range(len(xwerte)):
        eins += 1 / (yfehler[i]**2)
        
    return x,x2,y,xy,eins                        #gebe alle Werte zurück
    #return rundung(x), rundung(x2), rundung(y), rundung(xy), rundung(eins) #Ausgabe der Werte mit Rundung

def bestimmt(xwerte,ywerte,yfehler,a,b):
    yquer = support(xwerte,ywerte,yfehler)[2]/support(xwerte,ywerte,yfehler)[4]   #Berechnung von yquer
    zähler=0                                                                      #Zähler von R
    for i in range(len(ywerte)):                                                  #Summe über die
        zähler += ((ywerte[i]-a-b*xwerte[i])/(yfehler[i]))**2                     #Abweichungen von der Ausgleichsgeraden
    nenner=0                                                                      #Nenner von R
    for i in range(len(ywerte)):                                                  #Summe über die
        nenner += ((ywerte[i]-yquer)/(yfehler[i]))**2                             #Abweichungen vom Mittelwert yquer
    
    #return (1-zähler/nenner),zähler/(len(xwerte)-2)                               #R und (chi^2)/(n-2)
    return rundung((1-zähler/nenner)*100,zwischen=True), rundung(zähler/(len(xwerte)-2),zwischen=True)

def kern(xwerte,ywerte,yfehler):
    """Berechnet a, b, Delta a und Delta b."""
    
    (x,x2,y,xy,eins)=support(xwerte,ywerte,yfehler)  #erhält Hilfssummen aus support
    S = eins*x2-x**2                                 #Determinante der Koeffizientenmatrix S
    a=(x2*y-x*xy)/S                                  #y-Achsenabschnitt
    b=(eins*xy-x*y)/S                                #Steigung
    da=np.sqrt(x2/S)                                 #Fehler des y-Achsenabschnitts
    db=np.sqrt(eins/S)                               #Fehler der Steigung
    
    mux,muy=x/eins,y/eins
    #print(x,eins,len(xwerte))
    #print(mux,muy)
    
    return (a,da),(b,db),(mux,muy)
    #return rundung(a,da,zwischen=False), rundung(b,db,zwischen=False),(mux,muy)




#%%  Test der beiden Funktionen 

# Start mit lmfit routine (für evtl Zufallszahlen)

show_uncert=1
out,fig=linear_fit(show_sigenv=show_uncert) # picks random values
xwerte,ywerte,yfehler = out.x,out.y,out.y_error 


#out,fig=linear_fit(xwerte-mux,ywerte-muy,yfehler,show_sigenv=show_uncert) # shifts center of x and y back to 0

#out.plot_residuals()
#plt.show()


# Modifizierte Ausgabe der GP - Routine : Kann so von Studenten übernommen werden

if not 'fig' in locals():
    fig=plt.figure(figsize=(6,4),dpi=200)


xwerte,ywerte,yfehler = out.x,out.y,out.y_error    # Messwerte hinzufügen
(a,da),(b,db),(mux,muy)=kern(xwerte,ywerte,yfehler) 
R2,s2=bestimmt(xwerte,ywerte,yfehler,a,b)     #Bestimmtheitsmaß und Varianz erhalten


#corr=out.covar[0,1] # nimmt die covarianz aus lmfit um die Grenzgeraden zu f


xkoords=np.linspace(xwerte[0],xwerte[-1],100) #x-Wertliste für das Diagramm
#Messpunkte mit Fehlerbalken
plt.errorbar(xwerte,ywerte,yerr=yfehler,fmt="o",color="black",capsize=4,capthick=2,label="Datenpunkte")
plt.plot(xkoords,a+b*xkoords,color="blue",label="Ausgleichsgerade")     #Ausgleichsgerade
if show_uncert:
    error=db*(xkoords-mux)
    plt.plot(xkoords,a+b*xkoords-error,color="red")                       #obere Grenzgerade
    plt.plot(xkoords,a+b*xkoords+error,color="red",label="Grenzgeraden")  #untere Grenzgerade
    #error=np.abs((db*(xkoords-mux))**2+da**2-2*da*db*(xkoords-mux)*corr)**0.5
    #plt.plot(xkoords,a+b*xkoords+(db*(xkoords-mux))**2,color="green") 
    #plt.plot(xkoords,a+b*xkoords-error,color="green")   
    #plt.plot(xkoords,a+b*xkoords+error,color="green")
    error=((db*(xkoords-mux))**2+da**2-(db*mux)**2)**0.5
    plt.plot(xkoords,a+b*xkoords-error,color="cyan",ls='dashed')   
    plt.plot(xkoords,a+b*xkoords+error,color="cyan",ls='dashed',label='Uncertainty')  
plt.legend(loc="best")                                                  #Legende erstellen
#plt.title("Probemessung")                                               #Titel erstellen
plt.xlabel("x")                                                         #Beschriftung der x-Achse
plt.ylabel("y")                                                         #Beschriftung der y-Achse
plt.show()                                                              #fertiges Diagramm zeigen


(r_a,r_da),(r_b,r_db)=rundung(a,da,zwischen=True), rundung(b,db,zwischen=True)

print("Der y-Achsenabschnitt beträgt",r_a,"\u00B1",r_da,".")
print("Die Geradensteigung beträgt",r_b,"\u00B1",r_db,".")
print("Das Bestimmtheitsmaß R^2 beträgt",R2,".")
print("Die Varianz beträgt",s2,".")


    




