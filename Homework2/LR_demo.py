import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def lin_reg(x,y): 
    #number of observations / points
    n = np.size(x)

    #mean of x and y vector
    m_x, m_y, = np.mean(x) , np.mean(y)

    #calcualting cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    #calculating regression coefficiants
    beta = SS_xy / SS_xx
    alpha = m_y - beta*m_x

    return (alpha, beta)

def plot_lin_reg_model(x,y,a,b): 
    # plotting the actual points as a scatter plot
    plt.scatter(x,y,color="m", marker = "o", s=30)

    #predicted response vector
    # y_pred = alpha + beta*x
    y_pred = a + b*x


    #plotting the regresiion line
    plt.plot(x,y_pred, color='g')

    #putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    #function to show plot
    plt.show()

def main():

    #observations
    x=np.array([4,6,10,12])
    y=np.array([3.0,5.5,6.5,9.0])

    #estimating coefficients
    a,b = lin_reg(x,y)
    print("Estimated coefficients:\n alpha (slope intercept= {}     \n beta (slope) = {})".format(a,b))

    #plotting regression line
    plot_lin_reg_model(x,y,a,b)

    # Compare with sklearn 
    X=np.array([4,6,10,12])
    Y=np.array([3.0,5.5,6.5,9.0])
    XX = np.reshape(X,(-1,1))
    reg = LinearRegression().fit(XX,Y)

    # Coefficient of determination 
    # c_det=reg.score(XX,Y)
    #print("Estimated Coefficient of determinatino={}".format(c_det))


    #estimating coefficients
    print("Estimated coefficients:\n alpha (slope intercept) = {}   \n beta (slope) = {}".format(reg.intercept_, reg.coef_))

    #predict a new data point 
    #reg.predict(np.array([[3,5]]))

    new_x=3
    new_y= reg.predict(np.reshapre(new_x, (-1,1)))
    print("For new x = {} the estimated y prediction = {}". format(new_x, new_y))

    #plotting regression line 
    plot_lin_reg_model(np.append(x,new_x), np.append(y,new_y), reg.intercept_, reg.coef_)

if __name__ == "__main__":
    main()