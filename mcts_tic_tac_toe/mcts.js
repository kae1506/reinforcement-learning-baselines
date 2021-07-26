class Node{
    constructor(parent, state){
        this.parent = parent
        this.state = state
    
        this.player = null
        this.children = {}

        this.value = 0
        this.visits = 0

        this.isExpanded = False
        this.isTerminal = False
        if (this.state.isTerminal() == True){
            this.isTerminal = True
        }
    }

    choose_node(exploration_constant){
        var best_ucb = -Infinity;
        var best_node = null
        var ucb = 0
        for(var child in Object.values(this.children)){
            if (child.visits > 0){
                ucb = child.value / child.visits + exploration_constant * Math.sqrt(Math.log(this.visits/child.visits))
            }else {
                ucb = Infinity;
            }

            if (usb > best_ucb){
                best_ucb = ucb
                best_node = child
            }

        }


        return best_node
    }
}

class Board{
    constructor(board){
        if (board !== null){
            this.state = [...board.state]
            this.p1 = board.p1  
        }
        else {
            this.state = ["#", "#", "#", "#", "#", "#", "#", "#", "#"]
            this.p1 = 1
        }
    }

    generate_states(state){
        if (state == null){
            state = this
        }

        var states = []

        for (var i = 0; i < 9; i++){
            if(state.state[i] == "#"){
                var board = Board(state)
                board.p1 = 3 - board.p1
                states.push(board.make_move(i))      
            }
        }

        return states
    }

    make_move(position){
        board = Board(this)

        board.state[position] = board.p1

        return board
    }

    is_tie(){
        for (var i in this.state){
            if (i == "#"){
                return false
            }
        }

        return true
    }

    check(state){
        if (state == null){
            state = this.state
        }

        for (var i = 0; i < 3; i++){
            if (state[i*3] == state[i*3 + 1] == state[i*3 + 2] && state[i*3] != '#'):
                return state[i*3]
        }

        for (var i = 0; i < 3; i++){
            if (state[i] == state[i+3] == state[i+6] && state[i] != '#'){
                return state[i]
            }
        }

        if (state[0] == state[4] == state[8] && state[0] != '#'){
            return state[0]
        }

        if (state[2] == state[4] == state[6] && state[2] != '#'){
            return state[2]
        }

        if (self.is_tie()){
            return 0

        }

        return 'False'
    }

    get_winner(){
        if (self.check() != 0 && self.check() != 1 && self.check() != 2){
            console.log('called get winner when not terminal')
        }

        return self.check()
    }

    is_terminal(){
        if (self.check() != 0 && self.check() != 1 && self.check() != 2){
            return true
        }

        return false
    }
}

class MCTS{
    constructor(iterations){
        this.iterations = iterations
        this.tree = null
    }

    search(starting_board, player){
        var opponent = 3-player
        this.tree = Node(null, starting_board)
        this.tree.player = opponent
    
        for (var iteration = 0; iteration < this.iterations; iteration++){
            var node = this.traverse_and_expand(this.tree);

            var score = this.rollout(node, opponent);

            this.backpropogate(node, score)
        }

        var winner_node = this.tree.choose_node(0)

        return winner_node.state
    }


    traverse_and_expand(node){
        while( !node.state.is_terminal() ){
            if (node.isExpanded){
               node = node.choose_node(2) 
            }else{
                return this.expand(node)
            }
        }

        return node
    }

    expand(node){
        var states = node.state.generate_states()

        for (var state in states){
            child = Node(node, state);

            node.children[state] = child
            node.children[state].player = 3 - node.player
        }

        node.isExpanded = True
        return states[Math.floor(Math.random() * states.length)];
    }

    rollout(node, opponent){
        var tempNode = Node(node.parent, node.state);
        tempNode.player = node.player
        if (tempNode.state.is_terminal()){
            var status = tempNode.state.get_winner();

            if (status == opponent){
                temp_node.parent.value = -100000
                return status
            }
        }

        var board = temp_node.state
        while (!board.is_terminal()){
            possible_states = board.generate_states();
            board = possible_states[Math.floor(Math.Random() * possible_states.length)];
        }
        
        return board.get_winner();
    }

    backpropogate(node, result){
        var temp_node = node
        while (temp_node != null){
            temp_node.visits += 1
            if (temp_node.player == result){
                temp_node.value += 10
            }
            temp_node = temp_node.parent
        }
    }
}

var mcts = MCTS()
var b = Board()
b.p1 = 2

var p = 1
while (!b.is_terminal()){
    b = mcts.search(b, p)

    console.log('\n\n');
    var table = [];
    for (var i=0; i < 3; i++){
        table.push([])
        for (var j=0; j < 3; j++){
            table[i].push(b.state[i*3 + j])
        }
    }
    console.table(table)

    p = 3-p
}