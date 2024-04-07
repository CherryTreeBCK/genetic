import numpy as np
from keras.datasets import mnist
import numpy as np
import cv2
import genetic

def main():
    
    # Load the MNIST data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Optionally, you can normalize the image pixel values to the range 0 to 1
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # Now, train_images and test_images are NumPy arrays with the MNIST data
    print("Training data shape:", train_images.shape)
    print("Test data shape:", test_images.shape)
    
    # Resize the first image to 280x280 pixels
    resized_image = cv2.resize(train_images[0], (280, 280))
    
    # # Display the resized image
    # cv2.imshow(f"Train[0] {train_images[0].shape}", resized_image)
    # cv2.waitKey(1000)  # Display the window for 1 seconds
    
    # Reshape the data for training
    train_x = train_images.reshape(train_images.shape[0], 28*28)
    test_x = test_images.reshape(test_images.shape[0], 28*28)
    train_y = np.zeros((train_labels.shape[0], 10))
    train_y[np.arange(train_labels.shape[0]), train_labels] = 1
    test_y = np.zeros((test_labels.shape[0], 10))
    test_y[np.arange(test_labels.shape[0]), test_labels] = 1

    # Create a population
    population = genetic.GeneticAlgorithm(network_size=[train_x.shape[1], 16, train_y.shape[1]], mr=0.1, pop_size=100, loss_fnc='log')
    
    print_every_x_generations = 1
    # Perform the genetic algorithm
    for i in range(1000):
        if i % print_every_x_generations == 0:
            print("Generation: ", i)
            best = population.find_best(test_x, test_y)
            pred_y = best.forward(test_x)
            print("Best Accuracy: ", best.acc(pred_y, test_y))
            print("Best Log Loss: ", best.log_loss(pred_y, test_y))
            print("Best MSE Loss: ", best.mse_loss(pred_y, test_y))
            
            # Adjusted canvas size to accommodate scaled images
            display_scale = 5
            display_image = np.zeros((28 * display_scale * 3, 28 * display_scale * 3), dtype=np.uint8)

            for j in range(9):
                x_offset = (j % 3) * 28 * display_scale
                y_offset = (j // 3) * 28 * display_scale

                # Resize using display_scale
                resized_image = cv2.resize(test_x[j].reshape(28, 28), (20 * display_scale, 20 * display_scale))
                resized_image = (resized_image * 255).astype(np.uint8)
                
                # Correct placement using the scaled dimensions
                display_image[y_offset:y_offset + 20 * display_scale, x_offset:x_offset + 20 * display_scale] = resized_image
                pred = np.argmax(pred_y[j])
                true = np.argmax(test_y[j])
                # Text placement might need to be adjusted for visibility with scaled images
                cv2.putText(display_image, f"P:{pred} T:{true}", (x_offset+14, y_offset + 22 * display_scale), cv2.FONT_HERSHEY_SIMPLEX, 0.1 * display_scale, (255, 255, 255), 1)


            # Display the composite image
            cv2.imshow("Predictions", display_image)
            cv2.waitKey(1)  # Wait indefinitely until a key is pressed

        population.make_babies(test_x, test_y)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

