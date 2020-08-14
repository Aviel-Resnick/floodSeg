'''
Aviel Resnick, 2019
Utility designed for the automated, supervised, or manual morphometry of images, particularly stent-implanted coronary arteries.

supervisedSeg.py - controls supervised segmentation (through Data Exctraction menu)
'''

import cv2
import numpy as np
from collections import namedtuple
from itertools import cycle
import ui

Point = namedtuple('Point', 'x, y')
compContours = []
contourColor = (255, 0, 0)

class SelectionWindow:
    #_displays = cycle(['selection', 'mask', 'applied mask'])
    _displays = cycle(['selection'])

    def __init__(self, name, image, connectivity=4):
        # general params
        self.name = name
        self._image = image
        self._h, self._w = image.shape[:2]
        
        if len(image.shape) == 3:
            self._channels = 3
        else:
            self._channels = 1
        
        self._selection = image.copy()

        for i in compContours:
            cv2.drawContours(self._selection, i, -1, color=contourColor, thickness=2)

        self._mask = 255*np.ones((self._h, self._w), dtype=np.uint8)
        self._applied_mask = image.copy()
        self._curr_display = next(self._displays)

        # parameters for floodfill
        self.connectivity = connectivity
        self._tolerance = (25,)*3
        self._seed_point = Point(0, 0)
        self._flood_mask = np.zeros((self._h+2, self._w+2), dtype=np.uint8)

    def _onchange(self, pos):
        self._tolerance = (pos,)*3
        self._magicwand()

    def _onclick(self, event, x, y, flags, param):
        if flags & cv2.EVENT_FLAG_LBUTTON:
            self._seed_point = Point(x, y)
            self._magicwand()

    def _magicwand(self):
        self._flood_mask[:] = 0
        flags = self.connectivity | 255 << 16   # bit shift
        flags |= cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
        flood_image = self._image.copy()
        cv2.floodFill(flood_image, self._flood_mask, self._seed_point, 0, self._tolerance, self._tolerance, flags)
        self._mask = self._flood_mask[1:-1, 1:-1].copy()
        self._update_window()

    def _drawselection(self):
        # find contours around mask
        self._selection = self._image.copy()
        
        # FOR PRODUCTION
        self._contours, _ = cv2.findContours(self._mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # FOR TESTING
        #_, self._contours, _ = cv2.findContours(self._mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        #print(self._contours)
        cv2.drawContours(self._selection, self._contours, -1, color=contourColor, thickness=2)
        for i in compContours:
            cv2.drawContours(self._selection, i, -1, color=contourColor, thickness=2)

    def _flip_displays(self):
        self._curr_display = next(self._displays)
        self._update_window()
        if self.verbose:
            print('Displaying %s' % self._curr_display)

    def _close(self):
        if self.verbose:
            print('Closing window')
        cv2.destroyWindow(self.name)

    def _update_window(self):
        if self._curr_display == 'selection':
            self._drawselection()
            cv2.imshow(self.name, self._selection)
        elif self._curr_display == 'mask':
            cv2.imshow(self.name, self._mask)
        elif self._curr_display == 'applied mask':
            self._applied_mask = cv2.bitwise_and(self._image, self._image, mask=self._mask)
            cv2.imshow(self.name, self._applied_mask)

    def show(self, verbose=False):
        # create window, event callbacks
        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self._onclick)
        cv2.createTrackbar('Tolerance', self.name, self._tolerance[0], 255, self._onchange)
        self.verbose = verbose

        # display the image and wait for a keypress or trackbar change
        for i in compContours:
            cv2.drawContours(self._image, i, -1, color=contourColor, thickness=2)

        cv2.imshow(self.name, self._image)


        while(True):
            k = cv2.waitKey() & 0xFF
            # q, esc, or space close the window
            if k == ord('q') or k == 27 or k == 32:
                self._close()
                print("Finish Pressed; Stopping")
                #print(compContours)
                compContours.clear()
                if self._contours and hasattr(SelectionWindow, "_contours"):
                    return((False, self._contours))
                else:
                    return(False, self._contours)
                break
            elif k == ord('a'):
                #self._close()
                print("A Pressed; Going Again")
                compContours.append(self._contours)
                #print(compContours)
                return((True, self._contours)) # go again
                break

    @property
    def mask(self):
        return self._mask

    @property
    def applied_mask(self):
        self._applied_mask = cv2.bitwise_and(self._image, self._image, mask=self._mask)
        return self._applied_mask

    @property
    def selection(self):
        self._drawselection()
        return self._selection

    @property
    def contours(self):
        self._drawselection()
        return self._contours

    @property
    def seedpt(self):
        return self._seed_point

if __name__ == '__main__':
    main()