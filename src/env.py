"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import cv2
from pygame import RLEACCEL
from pygame.image import load
from pygame.sprite import Sprite, Group, collide_mask
from pygame import Rect, init, time, display, mixer, transform, Surface
from pygame.surfarray import array3d
import torch
from random import randrange, choice
import numpy as np

mixer.pre_init(44100, -16, 2, 2048)
init()

scr_size = (width, height) = (600, 150)
FPS = 60
gravity = 0.6

black = (0, 0, 0)
white = (255, 255, 255)
background_col = (235, 235, 235)

high_score = 0

screen = display.set_mode(scr_size)
clock = time.Clock()
display.set_caption("T-Rex Rush")


def load_image(
        name,
        sizex=-1,
        sizey=-1,
        colorkey=None,
):
    fullname = os.path.join("assets/sprites", name)
    image = load(fullname)
    image = image.convert()
    if colorkey is not None:
        if colorkey is -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)

    if sizex != -1 or sizey != -1:
        image = transform.scale(image, (sizex, sizey))

    return (image, image.get_rect())


def load_sprite_sheet(
        sheetname,
        nx,
        ny,
        scalex=-1,
        scaley=-1,
        colorkey=None,
):
    fullname = os.path.join("assets/sprites", sheetname)
    sheet = load(fullname)
    sheet = sheet.convert()

    sheet_rect = sheet.get_rect()

    sprites = []

    sizey = sheet_rect.height / ny
    if isinstance(nx, int):
        sizex = sheet_rect.width / nx
        for i in range(0, ny):
            for j in range(0, nx):
                rect = Rect((j * sizex, i * sizey, sizex, sizey))
                image = Surface(rect.size)
                image = image.convert()
                image.blit(sheet, (0, 0), rect)

                if colorkey is not None:
                    if colorkey is -1:
                        colorkey = image.get_at((0, 0))
                    image.set_colorkey(colorkey, RLEACCEL)

                if scalex != -1 or scaley != -1:
                    image = transform.scale(image, (scalex, scaley))

                sprites.append(image)

    else:  #list
        sizex_ls = [sheet_rect.width / i_nx for i_nx in nx]
        for i in range(0, ny):
            for i_nx, sizex, i_scalex in zip(nx, sizex_ls, scalex):
                for j in range(0, i_nx):
                    rect = Rect((j * sizex, i * sizey, sizex, sizey))
                    image = Surface(rect.size)
                    image = image.convert()
                    image.blit(sheet, (0, 0), rect)

                    if colorkey is not None:
                        if colorkey is -1:
                            colorkey = image.get_at((0, 0))
                        image.set_colorkey(colorkey, RLEACCEL)

                    if i_scalex != -1 or scaley != -1:
                        image = transform.scale(image, (i_scalex, scaley))

                    sprites.append(image)

    sprite_rect = sprites[0].get_rect()

    return sprites, sprite_rect


def extractDigits(number):
    if number > -1:
        digits = []
        i = 0
        while (number / 10 != 0):
            digits.append(number % 10)
            number = int(number / 10)

        digits.append(number % 10)
        for i in range(len(digits), 5):
            digits.append(0)
        digits.reverse()
        return digits


def pre_processing(image, w=84, h=84):
    image = image[:300, :, :]
    # cv2.imwrite("ori.jpg", image)
    image = cv2.cvtColor(cv2.resize(image, (w, h)), cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("color.jpg", image)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("bw.jpg", image)

    return image[None, :, :].astype(np.float32)


class Dino():
    def __init__(self, sizex=-1, sizey=-1):
        self.images, self.rect = load_sprite_sheet("dino.png", 5, 1, sizex, sizey, -1)
        self.images1, self.rect1 = load_sprite_sheet("dino_ducking.png", 2, 1, 59, sizey, -1)
        self.rect.bottom = int(0.98 * height)
        self.rect.left = width / 15
        self.image = self.images[0]
        self.index = 0
        self.counter = 0
        self.score = 0
        self.isJumping = False
        self.isDead = False
        self.isDucking = False
        self.isBlinking = False
        self.movement = [0, 0]
        self.jumpSpeed = 11.5

        self.stand_pos_width = self.rect.width
        self.duck_pos_width = self.rect1.width

    def draw(self):
        screen.blit(self.image, self.rect)

    def checkbounds(self):
        if self.rect.bottom > int(0.98 * height):
            self.rect.bottom = int(0.98 * height)
            self.isJumping = False

    def update(self):
        if self.isJumping:
            self.movement[1] = self.movement[1] + gravity

        if self.isJumping:
            self.index = 0
        elif self.isBlinking:
            if self.index == 0:
                if self.counter % 400 == 399:
                    self.index = (self.index + 1) % 2
            else:
                if self.counter % 20 == 19:
                    self.index = (self.index + 1) % 2

        elif self.isDucking:
            if self.counter % 5 == 0:
                self.index = (self.index + 1) % 2
        else:
            if self.counter % 5 == 0:
                self.index = (self.index + 1) % 2 + 2

        if self.isDead:
            self.index = 4

        if not self.isDucking:
            self.image = self.images[self.index]
            self.rect.width = self.stand_pos_width
        else:
            self.image = self.images1[(self.index) % 2]
            self.rect.width = self.duck_pos_width

        self.rect = self.rect.move(self.movement)
        self.checkbounds()

        if not self.isDead and self.counter % 7 == 6 and self.isBlinking == False:
            self.score += 1

        self.counter = (self.counter + 1)


class Cactus(Sprite):
    def __init__(self, speed=5, sizex=-1, sizey=-1):
        Sprite.__init__(self, self.containers)
        self.images, self.rect = load_sprite_sheet("cacti-small.png", [2, 3, 6], 1, sizex, sizey, -1)
        self.rect.bottom = int(0.98 * height)
        self.rect.left = width + self.rect.width
        self.image = self.images[randrange(0, 11)]
        self.movement = [-1 * speed, 0]

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)

        if self.rect.right < 0:
            self.kill()


class Ptera(Sprite):
    def __init__(self, speed=5, sizex=-1, sizey=-1):
        Sprite.__init__(self, self.containers)
        self.images, self.rect = load_sprite_sheet("ptera.png", 2, 1, sizex, sizey, -1)
        self.ptera_height = [height * 0.82, height * 0.75, height * 0.60, height * 0.48]
        self.rect.centery = self.ptera_height[randrange(0, 4)]
        self.rect.left = width + self.rect.width
        self.image = self.images[0]
        self.movement = [-1 * speed, 0]
        self.index = 0
        self.counter = 0

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        if self.counter % 10 == 0:
            self.index = (self.index + 1) % 2
        self.image = self.images[self.index]
        self.rect = self.rect.move(self.movement)
        self.counter = (self.counter + 1)
        if self.rect.right < 0:
            self.kill()


class Ground():
    def __init__(self, speed=-5):
        self.image, self.rect = load_image("ground.png", -1, -1, -1)
        self.image1, self.rect1 = load_image("ground.png", -1, -1, -1)
        self.rect.bottom = height
        self.rect1.bottom = height
        self.rect1.left = self.rect.right
        self.speed = speed

    def draw(self):
        screen.blit(self.image, self.rect)
        screen.blit(self.image1, self.rect1)

    def update(self):
        self.rect.left += self.speed
        self.rect1.left += self.speed

        if self.rect.right < 0:
            self.rect.left = self.rect1.right

        if self.rect1.right < 0:
            self.rect1.left = self.rect.right


class Cloud(Sprite):
    def __init__(self, x, y):
        Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image("cloud.png", int(90 * 30 / 42), 30, -1)
        self.speed = 1
        self.rect.left = x
        self.rect.top = y
        self.movement = [-1 * self.speed, 0]

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()


class Scoreboard():
    def __init__(self, x=-1, y=-1):
        self.score = 0
        self.tempimages, self.temprect = load_sprite_sheet("numbers.png", 12, 1, 11, int(11 * 6 / 5), -1)
        self.image = Surface((55, int(11 * 6 / 5)))
        self.rect = self.image.get_rect()
        if x == -1:
            self.rect.left = width * 0.89
        else:
            self.rect.left = x
        if y == -1:
            self.rect.top = height * 0.1
        else:
            self.rect.top = y

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self, score):
        score_digits = extractDigits(score)
        self.image.fill(background_col)
        if len(score_digits) == 6:
            score_digits = score_digits[1:]
        for s in score_digits:
            self.image.blit(self.tempimages[s], self.temprect)
            self.temprect.left += self.temprect.width
        self.temprect.left = 0


class ChromeDino(object):
    def __init__(self):
        self.gamespeed = 5
        self.gameOver = False
        self.gameQuit = False
        self.playerDino = Dino(44, 47)
        self.new_ground = Ground(-1 * self.gamespeed)
        self.scb = Scoreboard()
        self.highsc = Scoreboard(width * 0.78)
        self.counter = 0

        self.cacti = Group()
        self.pteras = Group()
        self.clouds = Group()
        self.last_obstacle = Group()

        Cactus.containers = self.cacti
        Ptera.containers = self.pteras
        Cloud.containers = self.clouds

        self.retbutton_image, self.retbutton_rect = load_image("replay_button.png", 35, 31, -1)
        self.gameover_image, self.gameover_rect = load_image("game_over.png", 190, 11, -1)

        self.temp_images, self.temp_rect = load_sprite_sheet("numbers.png", 12, 1, 11, int(11 * 6 / 5), -1)
        self.HI_image = Surface((22, int(11 * 6 / 5)))
        self.HI_rect = self.HI_image.get_rect()
        self.HI_image.fill(background_col)
        self.HI_image.blit(self.temp_images[10], self.temp_rect)
        self.temp_rect.left += self.temp_rect.width
        self.HI_image.blit(self.temp_images[11], self.temp_rect)
        self.HI_rect.top = height * 0.1
        self.HI_rect.left = width * 0.73

    def step(self, action, record=False):  # 0: Do nothing. 1: Jump. 2: Duck
        reward = 0.1
        if action == 0:
            reward += 0.01
            self.playerDino.isDucking = False
        elif action == 1:
            self.playerDino.isDucking = False
            if self.playerDino.rect.bottom == int(0.98 * height):
                self.playerDino.isJumping = True
                self.playerDino.movement[1] = -1 * self.playerDino.jumpSpeed

        elif action == 2:
            if not (self.playerDino.isJumping and self.playerDino.isDead) and self.playerDino.rect.bottom == int(
                    0.98 * height):
                self.playerDino.isDucking = True

        for c in self.cacti:
            c.movement[0] = -1 * self.gamespeed
            if collide_mask(self.playerDino, c):
                self.playerDino.isDead = True
                reward = -1
                break
            else:
                if c.rect.right < self.playerDino.rect.left < c.rect.right + self.gamespeed + 1:
                    reward = 1
                    break

        for p in self.pteras:
            p.movement[0] = -1 * self.gamespeed
            if collide_mask(self.playerDino, p):
                self.playerDino.isDead = True
                reward = -1
                break
            else:
                if p.rect.right < self.playerDino.rect.left < p.rect.right + self.gamespeed + 1:
                    reward = 1
                    break

        if len(self.cacti) < 2:
            if len(self.cacti) == 0 and len(self.pteras) == 0:
                self.last_obstacle.empty()
                self.last_obstacle.add(Cactus(self.gamespeed, [60, 40, 20], choice([40, 45, 50])))
            else:
                for l in self.last_obstacle:
                    if l.rect.right < width * 0.7 and randrange(0, 50) == 10:
                        self.last_obstacle.empty()
                        self.last_obstacle.add(Cactus(self.gamespeed, [60, 40, 20], choice([40, 45, 50])))

        # if len(self.pteras) == 0 and randrange(0, 200) == 10 and self.counter > 500:
        if len(self.pteras) == 0 and len(self.cacti) < 2 and randrange(0, 50) == 10 and self.counter > 500:
            for l in self.last_obstacle:
                if l.rect.right < width * 0.8:
                    self.last_obstacle.empty()
                    self.last_obstacle.add(Ptera(self.gamespeed, 46, 40))

        if len(self.clouds) < 5 and randrange(0, 300) == 10:
            Cloud(width, randrange(height / 5, height / 2))

        self.playerDino.update()
        self.cacti.update()
        self.pteras.update()
        self.clouds.update()
        self.new_ground.update()
        self.scb.update(self.playerDino.score)

        state = display.get_surface()
        screen.fill(background_col)
        self.new_ground.draw()
        self.clouds.draw(screen)
        self.scb.draw()
        self.cacti.draw(screen)
        self.pteras.draw(screen)
        self.playerDino.draw()

        display.update()
        clock.tick(FPS)

        if self.playerDino.isDead:
            self.gameOver = True

        self.counter = (self.counter + 1)

        if self.gameOver:
            self.__init__()

        state = array3d(state)
        if record:
            return torch.from_numpy(pre_processing(state)), np.transpose(
                cv2.cvtColor(state, cv2.COLOR_RGB2BGR), (1, 0, 2)), reward, not (reward > 0)
        else:
            return torch.from_numpy(pre_processing(state)), reward, not (reward > 0)


# if __name__ == "__main__":
#
#     dino = ChromeDino()
#     while True:
#         state, reward, done = dino.step(randint(0, 2))
