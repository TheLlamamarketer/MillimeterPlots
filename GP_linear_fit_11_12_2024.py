# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:48:27 2024

@author: cstrueber
"""
import numpy as np
import matplotlib.pyplot as plt

#%% Alle Funktionen der modifizierten GP Fitroutine

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


def plot_fit_GP(xwerte,ywerte,yfehler,figure='',xlabel='',ylabel=''):
    # xwerte,ywerte,yfehler sind notwendige inputs
    # figure erwartet das handle zu einer offenen Figure. Ansonsten machte es eine neue
    # xlabel und ylabel kann man direkt als Input definieren - oder auch nach Ausführen der Funktion!
    # alle weiteren Änderungen an der Figure können auch nach Ausführen der Funktion erfolgen
    
    if figure=='':
        figure=plt.figure(figsize=(6,4),dpi=200)
    else:
        plt.figure(figure)
        
    (a,da),(b,db),(mux,muy)=kern(xwerte,ywerte,yfehler) 
    R2,s2=bestimmt(xwerte,ywerte,yfehler,a,b)     #Bestimmtheitsmaß und Varianz erhalten

    xkoords=np.linspace(xwerte[0],xwerte[-1],100) #x-Wertliste für das Diagramm
    #Messpunkte mit Fehlerbalken
    plt.errorbar(xwerte,ywerte,yerr=yfehler,fmt="o",color="black",capsize=4,capthick=2,label="Datenpunkte")
    plt.plot(xkoords,a+b*xkoords,color="blue",label="Ausgleichsgerade")     #Ausgleichsgerade

    # bestimmt Grenzgeraden
    error=db*(xkoords-mux)   # translationsinvariante Definition für Grenzgeraden
    plt.plot(xkoords,a+b*xkoords-error,color="red")                       #obere Grenzgerade
    plt.plot(xkoords,a+b*xkoords+error,color="red",label="Grenzgeraden")  #untere Grenzgerade
    # bestimmt 1-sigma-Umgebung
    # wahrscheinlich wäre eine Students-Verteilung sinnvoller
    students=1
    if students:
        stud_fac=np.sqrt(xwerte.size/(xwerte.size-2)) # Varianz der Students-Verteilung für standardnormalverteilte X_i
    else:
        stud_fac=1
    
    error=stud_fac*((db*(xkoords-mux))**2+(da**2-(db*mux)**2))**0.5     # zweiter Summand ist y-Fehler bei (mux,muy) anstatt bei x=0
    plt.plot(xkoords,a+b*xkoords-error,color="cyan",ls='dashed')   
    plt.plot(xkoords,a+b*xkoords+error,color="cyan",ls='dashed',label='1-$\sigma$-Umgebung')  


    plt.legend(loc="best")                                                  #Legende erstellen
    #plt.title("Probemessung")                                               #Titel erstellen
    if xlabel=='':                                                        
        plt.xlabel('xlabel (units)')                                            #Beschriftung der x-Achse
    else:
        plt.xlabel(xlabel)
    if ylabel=='':                 
        plt.ylabel('ylabel (units)')                                            #Beschriftung der y-Achse
    else:
        plt.ylabel(ylabel)
        
    (r_a,r_da),(r_b,r_db)=rundung(a,da,zwischen=True), rundung(b,db,zwischen=True)
    print("Der y-Achsenabschnitt beträgt",r_a,"\u00B1",r_da,".")
    print("Die Geradensteigung beträgt",r_b,"\u00B1",r_db,".")
    print("Das Bestimmtheitsmaß R^2 beträgt",R2,".")
    print("Die Varianz beträgt",s2,".")
    return figure,a,da,b,db  # gibt figure handle und nicht gerundete Fitparameter aus


#%% Modifizierte Ausgabe der GP - Routine : Kann so von Studenten übernommen werden

# Messwerte ersetzen und ausführen
xwerte=np.array([1,2,3,4,5,6,7,8,9])
ywerte=np.array([4,4.08,4.25,4.5,4.6,4.75,4.86,5.01,5.09])
yfehler=.05*np.ones(9)

fig=plot_fit_GP(xwerte,ywerte,yfehler)
plt.xlabel('newxlabel')

plt.show()

#%% Test mit Zufallszahlen

nr_points=20

x=np.linspace(100,200,nr_points)
#x=x-np.mean(x)
y_error=0.1*np.ones(np.size(x)) # stddev
#y_error=10*np.arange(np.size(x))+0.01
offset=10
slope=0.03
y=offset+np.random.normal(0, y_error, size=x.size)+x*slope 

fig,a,da,b,db=plot_fit_GP(x,y,y_error)
plt.show()

# Test Residuum

fig,a,da,b,db=plot_fit_GP(x,y-b*x-a,y_error)
plt.plot(x,y*0,':') # eine Linie auf der Null  gepunktet
plt.show()

