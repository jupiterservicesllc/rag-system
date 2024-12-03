class Game2048 {
    constructor() {
        this.board = Array(4).fill().map(() => Array(4).fill(0));
        this.score = 0;
        this.gameBoard = document.getElementById('game-board');
        this.scoreElement = document.getElementById('score');
        this.initializeGame();
    }

    initializeGame() {
        this.addRandomTile();
        this.addRandomTile();
        this.renderBoard();
        this.setupEventListeners();
    }

    setupEventListeners() {
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case 'ArrowUp': this.move('up'); break;
                case 'ArrowDown': this.move('down'); break;
                case 'ArrowLeft': this.move('left'); break;
                case 'ArrowRight': this.move('right'); break;
            }
        });
    }

    addRandomTile() {
        const emptyCells = [];
        for (let r = 0; r < 4; r++) {
            for (let c = 0; c < 4; c++) {
                if (this.board[r][c] === 0) {
                    emptyCells.push({r, c});
                }
            }
        }

        if (emptyCells.length > 0) {
            const {r, c} = emptyCells[Math.floor(Math.random() * emptyCells.length)];
            this.board[r][c] = Math.random() < 0.9 ? 2 : 4;
        }
    }

    move(direction) {
        let moved = false;
        const rotatedBoard = this.rotateBoard(direction);
        
        for (let r = 0; r < 4; r++) {
            const row = rotatedBoard[r].filter(val => val !== 0);
            for (let c = 0; c < row.length - 1; c++) {
                if (row[c] === row[c + 1]) {
                    row[c] *= 2;
                    this.score += row[c];
                    row.splice(c + 1, 1);
                    moved = true;
                }
            }
            
            while (row.length < 4) {
                row.push(0);
            }
            
            rotatedBoard[r] = row;
        }

        this.board = this.unrotateBoard(rotatedBoard, direction);
        
        if (moved) {
            this.addRandomTile();
            this.renderBoard();
            this.updateScore();
        }
    }

    rotateBoard(direction) {
        let rotated = JSON.parse(JSON.stringify(this.board));
        
        switch(direction) {
            case 'left': return rotated;
            case 'right': 
                return rotated.map(row => row.reverse());
            case 'up':
                return rotated[0].map((val, index) => rotated.map(row => row[index]));
            case 'down':
                return rotated[0].map((val, index) => 
                    rotated.map(row => row[index]).reverse()
                );
        }
    }

    unrotateBoard(board, direction) {
        switch(direction) {
            case 'left': return board;
            case 'right': 
                return board.map(row => row.reverse());
            case 'up':
                return board[0].map((val, index) => 
                    board.map(row => row[index])
                );
            case 'down':
                return board[0].map((val, index) => 
                    board.map(row => row[index]).reverse()
                );
        }
    }

    renderBoard() {
        this.gameBoard.innerHTML = '';
        for (let r = 0; r < 4; r++) {
            for (let c = 0; c < 4; c++) {
                const tile = document.createElement('div');
                tile.classList.add('tile');
                
                if (this.board[r][c] !== 0) {
                    tile.textContent = this.board[r][c];
                    tile.classList.add(`tile-${this.board[r][c]}`);
                }
                
                this.gameBoard.appendChild(tile);
            }
        }
    }

    updateScore() {
        this.scoreElement.textContent = this.score;
    }
}

// Initialize the game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new Game2048();
});
