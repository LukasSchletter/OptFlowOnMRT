import matplotlib.pyplot as plt
from PIL import Image

def plot_bilder_nebeneinander(image_paths):
    # Anzahl der Bilder
    num_bilder = len(image_paths)
    
    # Erstellen einer Abbildung und Subplots
    fig, axes = plt.subplots(1, num_bilder, figsize=(15, 5))
    
    # Sicherstellen, dass axes immer eine Liste ist, auch bei nur einem Bild
    if num_bilder == 1:
        axes = [axes]

    # Bilder laden und plotten
    for i, image_path in enumerate(image_paths):
        # Bild laden
        bild = Image.open(image_path)
        
        # Bild im Subplot anzeigen
        axes[i].imshow(bild)
        axes[i].axis('off')  # Achsen ausblenden

    # Zeige das Plot
    plt.tight_layout()
    plt.show()

def plot_side_by_side(path_1, path_2):
    for number in range(952,976,1):
        print(number)
        f, axarr = plt.subplots(1,2)
        img1 = mpimg.imread('movie_folder/raft_reg/' + str(number) + '.png')
        img2 = mpimg.imread('movie_folder/nonreg_raft/' + str(number) + '.png')
        #imgplot = plt.imshow(img)
        #plt.show()
        axarr[0].imshow(img1)
        axarr[1].imshow(img2)
        plt.savefig('movie_raft/vergleich/' + str(number) + '.png')


if __name__ == "__main__":
     #difference_reg_normal()
     plot_side_by_side('movie_folder/nonreg_raft', 'movie_folder/raft_reg')

     # Beispielaufruf der Funktion
     image_paths = [
        'bild1.png',  # Ersetze dies mit den Pfaden zu deinen Bildern
        'bild2.png',
        'bild3.png',
        'bild4.png'
     ]

     plot_bilder_nebeneinander(image_paths)