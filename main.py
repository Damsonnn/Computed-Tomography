import datetime
import math
import os.path
import cv2 as cv2  
import numpy as np
import pydicom
import skimage.color
import skimage.draw
import skimage.transform


class Configuration:
    def __init__(self, delta, n, l):
        self.n = n
        self.l = l  # degrees
        self.delta_alfa = delta  # degrees
        self.r = 1  # initial, 2*max dimension for each picture coordinate
        self.aspect_ratio = 1
        # self.tomograf_folder = 'tomograf/'
        # self.data_tomograf = ['SADDLE_PE.JPG']

        # self.working_directory = self.tomograf_folder  # switching between tomography andy DICOM
        # self.working_data = self.data_tomograf  # switching between tomography andy DICOM

        self.img = []
        self.working_img = []
        self.result = []
        self.added = []
        self.img_center = np.array([0, 0])
        self.img_dim = [0, 0]
        self.pads = [0, 0]
        self.sinogram = []
        self.sin_cp = []  # for animation

        self.detectors = []
        self.emitter = np.array([0, self.r])
        self.density = []

        self.M = np.array([[math.cos(math.radians(self.delta_alfa)), -math.sin(math.radians(self.delta_alfa))],
                           [math.sin((math.radians(self.delta_alfa))), math.cos(math.radians(self.delta_alfa))]])

        self.kernel = []
        return

    def get_detectors_coordinate(self):
        self.detectors = np.empty(shape=[1, 2])
        for i in range(self.n):
            y = self.r * math.cos(
                (math.radians(0)) + math.pi - math.radians(self.l / 2) + i * math.radians(self.l) / (
                        self.n - 1))
            x = self.r * math.sin(
                (math.radians(0)) + math.pi - math.radians(self.l / 2) + i * math.radians(self.l) / (
                        self.n - 1))
            self.detectors = np.append(self.detectors, [[x, y]], axis=0)
        self.detectors = self.detectors[1:]

    def shift_coordinates_to_picture_center(self):
        self.emitter += self.img_center
        for d in self.detectors:
            d += self.img_center
        return

    def shift_coordinates_to_main(self):
        self.emitter -= self.img_center
        for d in self.detectors:
            d -= self.img_center
        return

    def read_picture(self, dicom_picture):

        # self.img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.img = dicom_picture
        h, w = self.img.shape
        self.aspect_ratio = w/h
        display_img(self.img, "Original", self.aspect_ratio)

        r = math.ceil(math.sqrt(2) * math.ceil(max(h, w) / 2))
        a = 2 * r
        pad_h = (a - h) // 2
        pad_w = (a - w) // 2
        self.pads = [pad_h, pad_w]
        self.img = np.pad(self.img, ((pad_h, pad_h), (pad_w, pad_w)), 'constant')

        print(f'Original picture dimensions: {w}x{h}')

        h, w = self.img.shape
        r = w / 2
        self.img_dim = [w, h]
        print(f'Adjusted and resized picture dimensions: {w}x{h}')
        # self.r = min(int(w / 2), int(h / 2)) - 1
        self.r = r - 1
        self.emitter[1] = self.r
        print("Radius set to ", self.r)
        self.img_center[0], self.img_center[1] = float(w / 2), float(h / 2)
        print(f'Image center set to {self.img_center[0]} {self.img_center[1]}')
        self.working_img = np.copy(self.img)
        # self.working_img = cv2.cvtColor(self.working_img, cv2.COLOR_GRAY2RGB)
        self.density = np.zeros_like(self.img)
        return self.img

    def rotate_all(self):
        """
        Rotates emitter and detectors coordinates
        """
        self.emitter = self.M.dot(self.emitter)
        for i in range(len(self.detectors)):
            self.detectors[i] = self.M.dot(self.detectors[i])

    def visualise(self):
        self.working_img = np.copy(self.img)
        self.working_img = cv2.cvtColor(self.working_img, cv2.COLOR_GRAY2RGB)
        self.working_img = cv2.circle(self.working_img, (self.img_center[0], self.img_center[1]), self.r, (0, 0, 255))

        self.working_img = cv2.circle(self.working_img, (int(self.emitter[0]), int(self.emitter[1])), 2, (0, 255, 0),
                                      -1)
        for p in self.detectors:
            self.working_img = cv2.circle(self.working_img, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
            self.working_img = cv2.line(self.working_img, (int(self.emitter[0]), int(self.emitter[1])),
                                        (int(p[0]), int(p[1])), (0, 255, 0))

    def setKernel(self, kernel_len):
        self.kernel = np.zeros(kernel_len)

        self.kernel[kernel_len // 2] = 1
        for i in range(0, kernel_len // 2):
            if i % 2 != 0:
                self.kernel[kernel_len // 2 + i] = -4 / (np.pi ** 2) / (i ** 2)
                self.kernel[kernel_len // 2 - i] = -4 / (np.pi ** 2) / (i ** 2)

    def display(self, only_result, wait_time):
        # self.visualise()
        # cv2.imshow("Working picture", self.working_img)
        # MAX_SIZE = 600
        # cv2.imshow("original image", skimage.transform.resize(self.img, (MAX_SIZE, MAX_SIZE), anti_aliasing=True))
        

        if not only_result:
            sin_cp = self.sin_cp.copy()
            cv2.normalize(sin_cp, sin_cp, 0, 1, cv2.NORM_MINMAX)
            display_img(sin_cp, 'Sinogram')
        
        res_cp = self.result.copy()
        res_cp = res_cp / self.added
        cv2.normalize(res_cp, res_cp, 0, 1, cv2.NORM_MINMAX)
        display_img(res_cp, 'Working...')

        if wait_time == 0:
            print("Press any key to continue...")
        cv2.waitKey(wait_time)

    # def create_sinogram(self):


def display_img(img, title, aspect_ratio = 1):
    MAX_SIZE = 600
    cv2.imshow(title, skimage.transform.resize(img, (MAX_SIZE, MAX_SIZE * aspect_ratio // 1), anti_aliasing=True))
    cv2.waitKey(1)


def convolve(sinogram, conf):
    sin_cp = sinogram.copy()
    for i in range(int(360 // conf.delta_alfa)):
        sin_cp[i, :] = np.convolve(sin_cp[i, :], conf.kernel, mode='same')
    return sin_cp


def run_tomography(path, delta, n, l, animation, filtered):
    # Dicom
    dc = pydicom.dcmread(path)
    print("Dicom file: ")
    print(dc)
    img = dc.pixel_array

    print("\nStart processing")

    conf = Configuration(delta, n, l)
    conf.read_picture(img)
    iterations = int(360 / conf.delta_alfa)
    animation_step = iterations // 18
    conf.result = np.zeros(shape=[conf.img_dim[1], conf.img_dim[0]])
    conf.added = np.zeros(shape=[conf.img_dim[1], conf.img_dim[0]])
    conf.sinogram = np.zeros(shape=[iterations, conf.n])
    conf.sin_cp = conf.sinogram.copy()

    conf.display(False, 1)

    print("Processing...")
    conf.get_detectors_coordinate()

    for i in range(iterations):
        conf.shift_coordinates_to_picture_center()
        for d in range(conf.n):
            cords = skimage.draw.line(int(conf.emitter[1]), int(conf.emitter[0]), int(conf.detectors[d][1]),
                                        int(conf.detectors[d][0]))
            sum = np.sum(conf.img[cords])
            conf.sinogram[i][d] = sum

        conf.shift_coordinates_to_main()
        conf.rotate_all()

    # splot
    if filtered:
        conf.setKernel(21)
        conf.sin_cp = convolve(conf.sinogram, conf)
    else:
        conf.sin_cp = conf.sinogram.copy()

    for i in range(iterations):
        conf.shift_coordinates_to_picture_center()
        for d in range(conf.n):
            cords = skimage.draw.line(int(conf.emitter[1]), int(conf.emitter[0]), int(conf.detectors[d][1]),
                                        int(conf.detectors[d][0]))
            conf.result[cords] += conf.sin_cp[i][d] / len(cords[0])
            conf.added[cords] += 1
        conf.shift_coordinates_to_main()

        if animation and (i % animation_step) == 0:
            conf.display(True, 1)

        conf.rotate_all()

    print(f'End of processing {path}')

    conf.display(False, 1)
    dc.BitsStored = 8
    dc.BitsAllocated = 8
    arr = conf.result.copy()
    arr = arr/conf.added
    cv2.normalize(arr, arr, 0, 255, cv2.NORM_MINMAX)
    arr = arr.astype(np.uint8)
    y0 = conf.pads[0]
    y1 = conf.img_dim[0] - conf.pads[0] 
    x0 = conf.pads[1]
    x1 = conf.img_dim[1] - conf.pads[1]
    print(x0, "-",x1, "-",y0, "-",y1)

    arr = arr[y0:y1, x0:x1]
    
    cv2.normalize(arr, arr, 0, 255, cv2.NORM_MINMAX)
    arr = arr.astype(np.uint8)

    display_img(arr, "Result", conf.aspect_ratio)
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    dc.Rows = arr.shape[0]
    dc.Columns = arr.shape[1]
    dc.PixelData = arr.tobytes()

    return dc


def main():
    np.seterr(invalid='ignore')
    DEFAULT_DELTA = 0.5
    DEFAULT_N = 901
    DEFAULT_L = 270
    while True:
        print("\n==================================================================================================")
        path = input("Insert path to file: ")
        while not os.path.exists(path):
            path = input("No such file in directory. Try again: ")
        print('In all questions below answer 0 is consider as "left default". Default value is presented in braces')
        print("---------------------------------------------------------------------------------------------------")
        delta = input(f"Insert delta_alpha parameter (in degrees) from range <0.01, 10> (default: {DEFAULT_DELTA}): ")
        while True:
            try:
                delta = float(delta)
                if delta != 0 and (delta < 0.01 or delta > 10):
                    raise IOError
            except ValueError:
                delta = input("This is not correct number. Try again: ")
            except IOError:
                delta = input("Value is not in range. Try again: ")
            else:
                if delta == 0:
                    delta = DEFAULT_DELTA
                break
        print("---------------------------------------------------------------------------------------------------")
        n = input(f"Insert number of detectors from range <22, 1001> (default: {DEFAULT_N}): ")
        while True:
            try:
                n = int(n)
                if n != 0 and (n < 22 or n > 1001):
                    raise IOError
            except ValueError:
                n = input("This is not correct number. Try again: ")
            except IOError:
                n = input("Value is not in range. Try again: ")
            else:
                if n == 0:
                    n = DEFAULT_N
                break
        print("---------------------------------------------------------------------------------------------------")
        l = input(f"Insert detectors span (in degrees) from range <10, 270> (default: {DEFAULT_L}): ")
        while True:
            try:
                l = float(l)
                if l != 0 and (l < 10 or l > 270):
                    raise IOError
            except ValueError:
                l = input("This is not correct number. Try again: ")
            except IOError:
                l = input("Value is not in range. Try again: ")
            else:
                if l == 0:
                    l = DEFAULT_L
                break
        print("---------------------------------------------------------------------------------------------------")
        DEFAULT_FILTER = True
        filtered = input(f"Do you want to use filter? (1 - Yes, 2 - No, default: Yes): ")
        while True:
            try:
                filtered = int(filtered)
                if filtered not in [0, 1, 2]:
                    raise IOError
            except ValueError:
                filtered = input("This is not correct number. Try again: ")
            except IOError:
                filtered = input("This is not correct answer. Try again: ")
            else:
                if filtered == 0:
                    filtered = DEFAULT_FILTER
                elif filtered == 1:
                    filtered = True
                elif filtered == 2:
                    filtered = False
                break
        print("---------------------------------------------------------------------------------------------------")
        DEFAULT_ANIMATION = False
        animation = input("Do you wish to see progress during work (1 - Yes, 2 - No, default: No): ")
        while True:
            try:
                animation = int(animation)
                if animation not in [0, 1, 2]:
                    raise IOError
            except ValueError:
                animation = input("This is not correct number. Try again: ")
            except IOError:
                animation = input("This is not correct answer. Try again: ")
            else:
                if animation == 0:
                    animation = DEFAULT_ANIMATION
                elif animation == 1:
                    animation = True
                elif animation == 2:
                    animation = False
                break
        print("==================================================================================================")
        print(
            f"Selected values:\nDelta_alpha: {delta}\nNumber of detectors: {n}\nDetectors span: {l}\nAnimation: {'Yes' if animation == 1 else 'No'}")

        dc = run_tomography(path, delta, n, l, animation, filtered)
        # dc = run_tomography('tomograf/CT_ScoutView.jpg', DEFAULT_DELTA, DEFAULT_N, DEFAULT_L, False)
        # dc = run_tomography('dicom_examples/Kwadraty2.dcm', DEFAULT_DELTA, DEFAULT_N, DEFAULT_L, False)

        answer = input("Do you wish to save dicom file? (1 - Yes, 2 - No, default: No): ")
        edit = False
        while True:
            try:
                answer = int(answer)
                if answer not in [0, 1, 2]:
                    raise IOError
            except ValueError:
                answer = input("This is not correct number. Try again: ")
            except IOError:
                answer = input("This is not correct answer. Try again: ")
            else:
                if answer == 1:
                    edit = True
                break
        if edit:
            answer = input("Insert patient's name: ")
            dc.PatientName = answer
            answer = input("Insert comment: ")
            dc.ImageComments = answer
            dc.InstanceCreationDate = datetime.date.today().strftime("%Y-%m-%d")
            print("\nNew file content: ")
            print(dc)

            new_file = os.path.basename(path).split(".")[0] + "_processed.dcm"
            print(f'Saved as new file: {new_file}')
            dc.save_as(new_file)

        again = input("Press 1 to start again, any key else: ")

        if again == '1':
            continue
        return


if __name__ == "__main__":
    main()

# Model stożkowy
# 1. Wyznaczamy pozycje emitera
# 2. Wyznaczamy pozycje n detektorow
# 3. Algorytmem Bresenhama przechodzimy po lini od emitera do każdego detektora
#     * każdą linię zapisujemy jako kolumnę w sinogramie
# 4. Obracamy układ współrzędnych emitera i detektorów o delta alfa
#     * każdy obrót (iterację) zapisujemy jako wiersz w sinogramie
# 5. Powtarzamy kroki 3-4
# 6. Normalizujemy sinogram
# ------------------------------------------------------------------------------
# Algorytm wstecznej projekcji
# 7. Powtarzamy proces na sinogramie aby otrzymać obraz oryginalny