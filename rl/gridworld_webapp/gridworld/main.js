import './style.css'
import Phaser from 'phaser'

//https://www.youtube.com/watch?v=0qtg-9M3peI&t=984s

const sizes = {
  width: 500,
  height: 500
}

const speedDown = 300

class GameScene extends Phaser.Scene {
  constructor(){ 
    super('scene-game')
    this.player
    this.cursor
    this.playerSpeed = speedDown + 50
  }

  //preload assets
  preload() {
    this.load.image('background', '/assets/bg.png')
    this.load.image('basket', '/assets/basket.png')
    this.load.image('apple', '/assets/apple.png')

  }

  create(){
    this.add.image(0,0,"background").setOrigin(0,0)
    this.player = this.physics.add.image(0,400,"basket").setOrigin(0,0)
    this.player.setImmovable = true
    this.player.body.allowGravity = false

    this.apple = this.physics.add.image(0,0,"apple").setOrigin(0,0)
    this.apple.setMaxVelocity(0, speedDown)

    this.cursor = this.input.keyboard.createCursorKeys()
    this.player.setCollideWorldBounds(true)

    this.textscore = this.add.text(0, 10, 'Score: 0', {fill: '#00000'})
  }

  getRandomX(){
    return Math.floor(Math.random() * 480)
  }

  update(){
    const {left, right} = this.cursor

    if(left.isDown){
      this.player.setVelocityX(-this.playerSpeed)
    } else if (right.isDown){
      this.player.setVelocityX(this.playerSpeed)
    } else {
      this.player.setVelocityX(0)
    }

    //check if apple falls out of world, reset its position
    if(this.apple.y >= sizes.height){
      this.apple.setY(0)
      this.apple.setX(100)
    }
  }
}


const config = {
  type: Phaser.WEBGL,
  width: sizes.width,
  height: sizes.height,
  canvas: gameCanvas,
  physics: {
    default: 'arcade',
    arcade: {
      gravity: {y: speedDown} ,
      debug:true
    }
  },
  scene: [GameScene]
}



const game = new Phaser.Game(config)
