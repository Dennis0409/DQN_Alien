import os
import random
import time
from enum import Enum
class obj(Enum):
    empty = 0
    wall = 1
    point = 2
    player = 3
    enemy = 4
    Exit = 5
class Act(Enum):
    W = 0
    A = 1
    S = 2
    D = 3
class moveobj():
    def __init__(self,x,y) :
        self.x=x
        self.y=y
class enemy(moveobj):
    def __init__(self, x, y,movpos):
        super().__init__(x, y)
        self.movpos=movpos
        self.movid=0
        self.buttom=obj.empty
        self.side = 1
    def move(self):
        self.movid+=self.side
        if(self.movid>=len(self.movpos)-1 or self.movid<=0):
            self.side=-self.side
        self.x = self.movpos[self.movid][0]
        self.y = self.movpos[self.movid][1]
dir = {Act.W:(-1,0),Act.A:(0,-1),Act.S:(1,0),Act.D:(0,1)}
show = {obj.empty:" ",obj.enemy:"E",obj.player:"P",obj.point:"⋆",obj.wall:"1",obj.Exit:"D"}
humaninp={'W':Act.W,'A':Act.A,'S':Act.S,'D':Act.D}
class game:
    # 清屏函數
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    # 顯示遊戲畫面
    def display_screen(self):
        self.clear_screen()
        for i in range(self.width):
            for j in range(self.heigth):
                print(show[obj(self.map[i][j])],end=' ')
            print()
        print("Score:" , self.score)
        if(self.end):
            print("gameover !")
    def __init__(self) :
        self.map=[
            [1,1,1],
            [2,0,0],
            [0,0,5]
                  ]
        self.player = moveobj(1,2)
        self.width = 3
        self.heigth = 3
        self.score = 0
        self.end = False
        self.enemylist=[enemy(2,1,[(2,1),(2,0)])]
        self.map[self.player.x][self.player.y] = 3
        for obj in self.enemylist:
            self.map[obj.x][obj.y] = 4
        self.display_screen()
            
        # initgame
    def get_state(self):
        return self.width
    def move(self,act):
        if(act in humaninp):
            act = humaninp[act]
        else:
            try:
                act=Act(act)
            except:
                return
            
        # 0W 1A 2S 3D
         
        newx=self.player.x
        newy=self.player.y
        if(act in dir):
            newx+=dir[act][0]
            newy+=dir[act][1]
            self.map[self.player.x][self.player.y]=obj.empty.value
            if(self.safe(newx,newy)):
                self.player.x=newx
                self.player.y=newy       
            self.player_move(newx,newy)
        if(self.end):
            self.display_screen()
            return
            
        for enemy in self.enemylist:
            self.map[enemy.x][enemy.y]=enemy.buttom.value
            enemy.move()
            enemy.buttom = obj(self.map[enemy.x][enemy.y])
            if(enemy.buttom==obj.player):
                self.gameover()
            self.map[enemy.x][enemy.y]=obj.enemy.value
        self.display_screen()
        return self.score
    def player_move(self,x,y):
        if(self.map[x][y]==obj.point.value):
            self.score+=10
        elif(self.map[x][y]==obj.Exit.value):
            self.score+=10
            self.gameover()
            return
        elif(self.map[x][y]==obj.enemy.value):
            self.gameover()
            return
        self.map[self.player.x][self.player.y]=obj.player.value
    def gameover(self):
        
        self.end=True
    def safe(self,x,y):
        if(not(x>=0 and x<=self.width and y>=0 and y<=self.heigth)):
            return False
        return  self.map[x][y]!=obj.wall.value
        

if __name__ == '__main__':
    gam=game()
    while(not gam.end):
        gam.move(input())
        
        