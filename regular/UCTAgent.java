package src.pas.hex.agents;

import edu.bu.pas.hex.Action;
import edu.bu.pas.hex.BoardView;
import edu.bu.pas.hex.GameView;
import edu.bu.pas.hex.Game;
import edu.bu.pas.hex.GenericGame;
import edu.bu.pas.hex.agents.ATPZSMCTSAgent;
import edu.bu.pas.hex.enums.Owner;
import edu.bu.pas.hex.tree.ATPZSNode;
import edu.bu.pas.hex.tree.ATPZSQNode;
import edu.bu.pas.hex.tree.Node;
import edu.bu.pas.hex.utils.Coordinate;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Regular-credit Hex agent: UCT chooses where to spend a limited number of simulations.
 * Every outcome is stored from this agent's point of view: +100 for a win, -100 for a loss.
 * The four stages of each iteration are selection, expansion, simulation, and backup.
 * This study copy has the same executable code as the regular submission.
 */
public class UCTAgent extends ATPZSMCTSAgent<ATPZSQNode> {
    // Assignment defaults: ordinary UCT with random rollouts and final choice by mean reward.
    // The optional bridge and tactical helpers below are inactive with these settings.
    protected double exploration = Math.sqrt(2);
    protected boolean bridgeRollouts = false;
    protected boolean mostVisited = false;
    protected boolean tacticalMoves = false;
    private ATPZSQNode rolloutLeaf;
    private Action firstRolloutAction;
    private static final int[] DX = {1, 1, 0, -1, -1, 0};
    private static final int[] DY = {-1, 0, 1, 1, 0, -1};

    // The course framework stores our team number, time allowance, and random generator.
    public UCTAgent(final int playerIdx, final long maxThinkingTimeInMs) {
        super(playerIdx, maxThinkingTimeInMs);
    }

    // Each real turn begins with a fresh search tree rooted at a copy of the current board.
    // MAX is our agent's turn, whether we play Team A or Team B. Child nodes alternate MAX and MIN.
    @Override
    public ATPZSQNode toNode(final GameView game) {
        return new ATPZSQNode(new GameView(game), ATPZSNode.PlayerType.MAXIMIZING_PLAYER);
    }

    // Selection balances estimated reward against uncertainty. We rank existing children by
    // signed average reward plus C * sqrt(log(parent samples) / action samples).
    private ATPZSQNode bestUCB(ATPZSQNode node) {
        ATPZSQNode best = null;
        double bestValue = Double.NEGATIVE_INFINITY;
        // Q-values use this agent's perspective at every node.
        boolean maximizing = node.getPlayerType() == ATPZSNode.PlayerType.MAXIMIZING_PLAYER;
        for(Node child : node.getChildList()) {
            Action action = child.getLastAction();
            int visits = node.getNumSamples(action);
            // A cached move with no direct samples must get a trial before we compare averages.
            double value = Double.POSITIVE_INFINITY;
            if(visits > 0) {
                double q = node.getAverageQValue(action);
                // Our player prefers high Q; the opponent prefers low Q. Flip only the Q
                // term at MIN nodes. Both players still get a positive exploration bonus.
                value = (maximizing ? q : -q)
                    + this.exploration * Math.sqrt(Math.log(Math.max(1, node.getNumSamples())) / visits);
            }
            if(best == null || value > bestValue) {
                best = (ATPZSQNode)child;
                bestValue = value;
            }
        }
        return best;
    }

    // Step 1: descend through fully expanded positions. Stop at a game-over position or
    // a position that still has a legal action without a cached child.
    @Override
    public ATPZSQNode selectArtificialLeaf(final ATPZSQNode rootNode) {
        ATPZSQNode node = rootNode;
        while(!node.isTerminal()) {
            if(node.getChildList().size() < node.getState().getNumActions()) return node;
            node = bestUCB(node);
        }
        return node;
    }

    // Step 2: expand one move. Each missing child is an unexplored candidate, so choose
    // uniformly among them. This tests for a cached child rather than a zero sample count:
    // the first rollout at a node can sample a move without creating its child.
    @Override
    public ATPZSQNode chooseChild(final ATPZSQNode artificialLeafNode) {
        if(artificialLeafNode.isTerminal()) return artificialLeafNode;
        Set<Action> expanded = new HashSet<>();
        for(Node child : artificialLeafNode.getChildList()) {
            expanded.add(child.getLastAction());
        }
        List<Action> untried = new ArrayList<>();
        for(Action action : artificialLeafNode.getState().getActions()) {
            if(!expanded.contains(action)) untried.add(action);
        }
        if(untried.isEmpty()) return bestUCB(artificialLeafNode);
        Action action = untried.get(this.getRandom().nextInt(untried.size()));
        return cachedChild(artificialLeafNode, action);
    }

    // The framework's getChild constructs a node with the new board, parent and incoming move.
    // We must explicitly cache it with addChild so later iterations reuse its statistics.
    private ATPZSQNode cachedChild(ATPZSQNode parent, Action action) {
        for(Node child : parent.getChildList()) {
            if(action.equals(child.getLastAction())) return (ATPZSQNode)child;
        }
        try {
            ATPZSQNode child = (ATPZSQNode)parent.getChild(action);
            parent.addChild(child);
            return child;
        } catch(Exception exception) {
            throw new IllegalStateException("Could not expand a legal action", exception);
        }
    }

    // Optional tactical shortcut: test each legal move for our immediate win first. If none
    // wins, give the opponent the next turn on a copy and look for an immediate winning cell
    // we can occupy to block. This checks a single reply, not a proof against multiple threats.
    private ATPZSQNode immediateMove(ATPZSQNode root, long deadline) {
        if(!(root.getState() instanceof GameView game)) return null;
        try {
            for(Action action : game.getActions()) {
                if(System.nanoTime() >= deadline) return null;
                GameView next = action.apply(game);
                if(next.isOver() && next.getTerminalUtility(this.getPlayerIdx()) > 0) {
                    ATPZSQNode child = cachedChild(root, action);
                    child.setTerminalUtility(100d);
                    return child;
                }
            }
            Game opponent = new Game(game);
            opponent.setIsCurrentlyTeamA(!game.getIsCurrentlyTeamA());
            GameView threat = new GameView(opponent);
            for(Action action : game.getActions()) {
                if(System.nanoTime() >= deadline) return null;
                Action reply = new Action(action.coordinate(), threat.getCurrentTeam());
                GameView next = reply.apply(threat);
                if(next.isOver() && next.getTerminalUtility(1 - this.getPlayerIdx()) > 0) return cachedChild(root, action);
            }
            return null;
        } catch(Exception exception) {
            throw new IllegalStateException("Could not check an immediate threat", exception);
        }
    }

    // Breadth-first search: seed the queue with friendly stones on the starting edge, then
    // follow neighboring friendly stones. A connects top to bottom; B connects left to right.
    private boolean connected(int[] cells, int dimension, int player) {
        boolean[] seen = new boolean[cells.length];
        int[] queue = new int[cells.length];
        int head = 0, tail = 0;
        for(int i = 0; i < dimension; i++) {
            int index = player == 0 ? i : i * dimension;
            if(cells[index] == player) {
                seen[index] = true;
                queue[tail++] = index;
            }
        }
        while(head < tail) {
            int index = queue[head++];
            int x = index % dimension, y = index / dimension;
            if((player == 0 ? y : x) == dimension - 1) return true;
            for(int direction = 0; direction < 6; direction++) {
                int nx = x + DX[direction], ny = y + DY[direction];
                if(nx < 0 || ny < 0 || nx >= dimension || ny >= dimension) continue;
                int next = ny * dimension + nx;
                if(!seen[next] && cells[next] == player) {
                    seen[next] = true;
                    queue[tail++] = next;
                }
            }
        }
        return false;
    }

    // A friendly goal edge can be one endpoint of a virtual bridge, just like a friendly stone.
    private boolean ownsOrEdge(int[] cells, int dimension, int x, int y, int player) {
        if(x >= 0 && y >= 0 && x < dimension && y < dimension) return cells[y * dimension + x] == player;
        return player == 0 ? x >= 0 && x < dimension && (y < 0 || y >= dimension)
            : y >= 0 && y < dimension && (x < 0 || x >= dimension);
    }

    // A bridge is a pair of endpoints that can connect through either of two shared cells.
    // If the opponent just occupied one shared cell, try to take the other one immediately.
    // Rotate the pattern around the last move; return -1 when no matching reply is available.
    protected int bridgeReply(int[] cells, int dimension, int last, int player) {
        if(last < 0) return -1;
        int x = last % dimension, y = last / dimension;
        for(int i = 0; i < 6; i++) {
            int first = i, middle = (i + 1) % 6, second = (i + 2) % 6;
            int ax = x + DX[first], ay = y + DY[first];
            int bx = x + DX[second], by = y + DY[second];
            int rx = x + DX[middle], ry = y + DY[middle];
            if(rx < 0 || ry < 0 || rx >= dimension || ry >= dimension) continue;
            if(ownsOrEdge(cells, dimension, ax, ay, player) && ownsOrEdge(cells, dimension, bx, by, player)
                && cells[ry * dimension + rx] == -1) return ry * dimension + rx;
        }
        return -1;
    }

    // Step 3: simulate from the expanded child. Pack (x, y) into y * dimension + x.
    // cells stores ownership (-1 empty, 0 A, 1 B). Only the first 'remaining' entries
    // in empty are available. positions maps an empty cell back to its slot for bridge replies.
    private double hexRollout(GameView game, Action lastAction) {
        BoardView board = game.getBoard();
        int dimension = board.getDim(), size = dimension * dimension;
        int[] cells = new int[size], empty = new int[size], positions = new int[size];
        int remaining = 0;
        for(int index = 0; index < size; index++) {
            Owner owner = board.getOwnerAt(new Coordinate(index % dimension, index / dimension));
            cells[index] = owner == Owner.UNOWNED ? -1 : owner == Owner.TEAM_A ? 0 : 1;
            if(cells[index] == -1) {
                positions[index] = remaining;
                empty[remaining++] = index;
            }
        }
        int player = game.getIsCurrentlyTeamA() ? 0 : 1;
        int last = lastAction == null ? -1 : lastAction.coordinate().y() * dimension + lastAction.coordinate().x();
        while(remaining > 0) {
            int reply = this.bridgeRollouts ? bridgeReply(cells, dimension, last, player) : -1;
            int choice = reply < 0 ? this.getRandom().nextInt(remaining) : positions[reply];
            int index = empty[choice];
            // Remove the chosen empty cell by swapping in the last available one. This avoids
            // shifting the array, and the reverse lookup follows the swapped cell.
            empty[choice] = empty[--remaining];
            positions[empty[choice]] = choice;
            cells[index] = player;
            // Save the first action that is legal FROM this leaf. Its incoming move has already
            // been applied and therefore cannot receive the leaf's outgoing action sample.
            if(this.firstRolloutAction == null) {
                this.firstRolloutAction = new Action(new Coordinate(index % dimension, index / dimension),
                    player == 0 ? Owner.TEAM_A : Owner.TEAM_B);
            }
            last = index;
            player = 1 - player;
        }
        // A completed Hex board has exactly one winner. A connecting chain cannot be broken
        // by filling more cells, so one BFS at the end gives the same winner as stopping early.
        // Filling the remaining cells cannot change a Hex winner.
        int winner = connected(cells, dimension, 0) ? 0 : 1;
        return winner == this.getPlayerIdx() ? 100d : -100d;
    }

    // A terminal leaf needs no simulation. Real Hex positions use arrays for speed; other
    // GenericGame states use their own legal-action and transition methods on a mutable copy.
    @Override
    public double getRolloutOutcome(final ATPZSQNode leafNode) {
        this.rolloutLeaf = leafNode;
        this.firstRolloutAction = null;
        if(leafNode.isTerminal()) {
            double utility = leafNode.getState().getTerminalUtility(this.getPlayerIdx());
            leafNode.setTerminalUtility(utility);
            return utility;
        }
        if(leafNode.getState() instanceof GameView game) return hexRollout(game, leafNode.getLastAction());
        GenericGame game = leafNode.getState().getMutableInstance();
        try {
            while(!game.isOver()) {
                List<Action> actions = new ArrayList<>(game.getActions());
                Action action = actions.get(this.getRandom().nextInt(actions.size()));
                if(this.firstRolloutAction == null) this.firstRolloutAction = action;
                game = game.apply(action);
            }
            return game.getTerminalUtility(this.getPlayerIdx());
        } catch(Exception exception) {
            throw new IllegalStateException("Could not complete rollout", exception);
        }
    }

    // Step 4: store the leaf's first rollout sample, then credit the actual path up to the root.
    // Example: X plays A from root, O plays B first in the rollout, and X wins. Store +100
    // under A at the root and B at the leaf. There is no direct root sample for other rollout moves.
    @Override
    public void backupStatistics(final ATPZSQNode leafNode, double rolloutOutcomeUtility) {
        if(leafNode.isTerminal()) leafNode.setTerminalUtility(rolloutOutcomeUtility);
        if(leafNode.getParent() != null) {
            Action action = this.rolloutLeaf == leafNode ? this.firstRolloutAction : null;
            if(action == null) action = leafNode.isTerminal() ? leafNode.getLastAction()
                : leafNode.getState().getActionIterator().next();
            // addSample increments both the node's sample count and the count for this action.
            // A terminal leaf has no outgoing action; its incoming move is used for that case.
            leafNode.addSample(action, rolloutOutcomeUtility);
        }
        this.rolloutLeaf = null;
        this.firstRolloutAction = null;
        ATPZSQNode child = leafNode;
        while(child.getParent() != null) {
            ATPZSQNode parent = (ATPZSQNode)child.getParent();
            // At the parent, the move entering this child is a legal outgoing action.
            // Credit exactly this edge; the outcome always stays in our agent's perspective.
            parent.addSample(child.getLastAction(), rolloutOutcomeUtility);
            child = parent;
        }
    }

    // Orchestrate one real decision. Repeat selection, expansion, rollout, and backup.
    // The returned node is a direct child of the root; its incoming action is the real move.
    @Override
    public ATPZSQNode treeSearch(final ATPZSQNode rootNode) {
        if(rootNode.isTerminal()) return rootNode;
        long budget = this.getMaxThiningTimeInMs();
        // Reserve about one third of the allowance (at least 5 ms). For 500 ms, start new
        // iterations for at most 334 ms. This is a soft cutoff, not a hard interrupt.
        long deadline = System.nanoTime() + Math.max(0, budget - Math.max(5, budget / 3)) * 1000000L;
        if(this.tacticalMoves && budget >= 10) {
            ATPZSQNode forced = immediateMove(rootNode, deadline);
            if(forced != null) return forced;
        }
        ATPZSQNode best = null;
        int iterations = 0;
        // Complete the entire iteration before checking time again so every sampled child
        // has consistent statistics. The reserve leaves room to finish and choose a move.
        while(System.nanoTime() < deadline && iterations < 200000 && !Thread.currentThread().isInterrupted()) {
            ATPZSQNode leaf = selectArtificialLeaf(rootNode);
            ATPZSQNode child = chooseChild(leaf);
            backupStatistics(child, getRolloutOutcome(child));
            iterations++;
            if(child.getParent() == rootNode && child.isTerminal() && child.getTerminalUtility() == 100d) return child;
        }
        double bestValue = Double.NEGATIVE_INFINITY;
        for(Node child : rootNode.getChildList()) {
            Action action = child.getLastAction();
            int visits = rootNode.getNumSamples(action);
            if(visits == 0) continue;
            // The regular agent uses average reward here. With wins at +100 and losses at
            // -100, this ranks moves by sampled win rate. The exploration bonus is for search.
            double value = this.mostVisited ? visits : rootNode.getAverageQValue(action);
            if(value > bestValue) {
                best = (ATPZSQNode)child;
                bestValue = value;
            }
        }
        // With no completed samples (for example, an extremely small budget), still return
        // a legal root child rather than a null move.
        if(best != null) return best;
        return rootNode.getChildList().isEmpty() ? chooseChild(rootNode) : (ATPZSQNode)rootNode.getChildList().get(0);
    }
}
