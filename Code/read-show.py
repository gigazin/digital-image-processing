import numpy as np
import cv2 as cv

def main():
    src = cv.imread('C:/opencv/sources/samples/data/fruits.jpg')
    if src is None:
        print('Image not found!', src)
        sys.exit(1)
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # Sobel
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])

    # Default
    grad_x = cv.filter2D(gray, -1, kernel_x)
    grad_y = cv.filter2D(gray, -1, kernel_y)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    # Flipped
    grad_x_flip = cv.flip(grad_x, 1)
    grad_y_flip = cv.flip(grad_y, 1)

    abs_grad_x_flip = cv.convertScaleAbs(grad_x_flip)
    abs_grad_y_flip = cv.convertScaleAbs(grad_y_flip)

    grad_flip = cv.addWeighted(abs_grad_x_flip, 0.5, abs_grad_y_flip, 0.5, 0)

    while True:
        ch = cv.waitKey()
        if ch == 27:
            break
        cv.imshow('Read and Show Image', grad)
        cv.imshow('Read and Show Image (Flipped)', grad_flip)

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
