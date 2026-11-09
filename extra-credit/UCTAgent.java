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
 * Extra-credit Hex agent: the same UCT loop with separate RAVE statistics.
 * Direct samples measure taking a move now; AMAF samples also use moves played later by
 * the same side. Later placements can help predict earlier placements in Hex, but timing
 * still matters, so the estimates are blended instead of counting AMAF as direct evidence.
 * This study copy has the same executable code as the extra-credit submission.
 */
public class UCTAgent extends ATPZSMCTSAgent<ATPZSQNode> {
    // Extra-credit defaults: add tactical checks, bridge replies, and RAVE to ordinary UCT.
    // Use visit count for the final move. C = 50 sets the bonus scale for rewards of +/-100.
    protected double exploration = 50;
    protected boolean bridgeRollouts = true;
    protected boolean mostVisited = true;
    protected boolean tacticalMoves = true;
    protected boolean useRAVE = true;
    // Larger equivalence values let AMAF keep more influence at a given direct sample count.
    // The prior adds eight zero-reward pseudo-samples when estimating the AMAF mean.
    protected double raveEquivalence = 300;
    protected int ravePrior = 8;
    // Parallel arrays store the cell and team for each simulated placement in this rollout.
    private int[] rolloutMoves;
    private int[] rolloutPlayers;
    private int rolloutLength;

    // Separate per-node statistics for All Moves As First (AMAF). For the side to move at
    // this node, visits[i] counts simulations containing its move at cell i in the suffix.
    // totals[i] sums those outcomes. These samples never increment the ordinary Q counts.
    private static class RaveStats extends Node.ExtraParams {
        final int[] visits;
        final double[] totals;

        RaveStats(int size) {
            this.visits = new int[size];
            this.totals = new double[size];
        }
    }
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
                double bonus = this.exploration * Math.sqrt(Math.log(Math.max(1, node.getNumSamples())) / visits);
                double weight = 0;
                if(this.useRAVE && node.getExtraParams() instanceof RaveStats stats) {
                    int index = action.coordinate().y() * ((GameView)node.getState()).getBoard().getDim()
                        + action.coordinate().x();
                    // AMAF treats a move played later by this same side as indirect evidence
                    // about playing it now. The added count pulls a sparse estimate toward 0.
                    int amafVisits = stats.visits[index] + this.ravePrior;
                    if(amafVisits > 0) {
                        // beta = m / (m + n + m*n/k), where m includes the prior, n counts
                        // direct trials and k is the equivalence setting. More direct trials
                        // lower beta, letting the actual outcomes of choosing this move dominate.
                        weight = amafVisits / (amafVisits + visits + (double)amafVisits * visits / this.raveEquivalence);
                        q = (1 - weight) * q + weight * stats.totals[index] / amafVisits;
                    }
                }
                // Q and AMAF both use our perspective. The opponent prefers low blended Q.
                // This variant also shrinks the exploration bonus by the direct weight (1 - beta).
                value = (maximizing ? q : -q) + (1 - weight) * bonus;
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

    // Union-find keeps each connected group of friendly stones under one representative.
    // Path halving shortens parent chains as we find that representative.
    private int find(int[] parent, int cell) {
        while(parent[cell] != cell) {
            parent[cell] = parent[parent[cell]];
            cell = parent[cell];
        }
        return cell;
    }

    // Link the two group representatives so later finds recognize one connected component.
    private void join(int[] parent, int first, int second) {
        parent[find(parent, first)] = find(parent, second);
    }

    // Join a stone to neighboring friendly stones and any friendly goal edges it touches.
    // The four virtual edge indexes after the cells represent A's two edges and B's two edges.
    private void connectStone(int[] cells, int[] parent, int dimension, int index) {
        int player = cells[index], size = cells.length;
        int x = index % dimension, y = index / dimension;
        int side = player == 0 ? y : x;
        if(side == 0) join(parent, index, size + 2 * player);
        if(side == dimension - 1) join(parent, index, size + 2 * player + 1);
        for(int direction = 0; direction < 6; direction++) {
            int nx = x + DX[direction], ny = y + DY[direction];
            if(nx >= 0 && ny >= 0 && nx < dimension && ny < dimension) {
                int next = ny * dimension + nx;
                if(cells[next] == player) join(parent, index, next);
            }
        }
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
        int[] parent = null;
        // Build connectivity for the stones already on the board before simulating new moves.
        if(this.useRAVE) {
            this.rolloutMoves = new int[remaining];
            this.rolloutPlayers = new int[remaining];
            parent = new int[size + 4];
            for(int i = 0; i < parent.length; i++) parent[i] = i;
            for(int i = 0; i < size; i++) if(cells[i] >= 0) connectStone(cells, parent, dimension, i);
        }
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
            if(this.useRAVE) {
                // Record the winning placement, then stop. Continuing after the win would
                // give AMAF credit for moves that could never occur in that game.
                this.rolloutMoves[this.rolloutLength] = index;
                this.rolloutPlayers[this.rolloutLength++] = player;
                connectStone(cells, parent, dimension, index);
                if(find(parent, size + 2 * player) == find(parent, size + 2 * player + 1)) {
                    return player == this.getPlayerIdx() ? 100d : -100d;
                }
            }
            last = index;
            player = 1 - player;
        }
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
        this.rolloutLength = 0;
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

    // Step 4: record ordinary Q along the actual path, plus separate AMAF samples for later
    // moves. If X plays A at root, O plays B in rollout, X plays C, and X wins, root Q credits A;
    // root AMAF credits A and C; the node after A credits B for O. All rewards stay +100 for X.
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
        // Start with only the rollout suffix. As we climb, prepend each incoming tree move
        // so the next parent sees all moves from its own position onward.
        boolean[][] later = null;
        int dimension = 0;
        if(this.useRAVE && leafNode.getState() instanceof GameView game) {
            dimension = game.getBoard().getDim();
            later = new boolean[2][dimension * dimension];
            for(int i = 0; i < this.rolloutLength; i++) later[this.rolloutPlayers[i]][this.rolloutMoves[i]] = true;
            updateRave(leafNode, later, rolloutOutcomeUtility);
        }
        ATPZSQNode child = leafNode;
        while(child.getParent() != null) {
            ATPZSQNode parent = (ATPZSQNode)child.getParent();
            // At the parent, the move entering this child is a legal outgoing action.
            // Credit exactly this edge; the outcome always stays in our agent's perspective.
            parent.addSample(child.getLastAction(), rolloutOutcomeUtility);
            if(later != null) {
                Action action = child.getLastAction();
                int player = action.owner() == Owner.TEAM_A ? 0 : 1;
                later[player][action.coordinate().y() * dimension + action.coordinate().x()] = true;
                updateRave(parent, later, rolloutOutcomeUtility);
            }
            child = parent;
        }
    }

    // AMAF only credits moves made by the side to move here. A Hex placement is permanent,
    // so a cell played in this suffix was empty at this node. The boolean row counts it once.
    private void updateRave(ATPZSQNode node, boolean[][] later, double outcome) {
        if(node.isTerminal()) return;
        GameView game = (GameView)node.getState();
        int player = game.getIsCurrentlyTeamA() ? 0 : 1;
        RaveStats stats;
        if(node.getExtraParams() instanceof RaveStats existing) stats = existing;
        else {
            stats = new RaveStats(later[player].length);
            node.setExtraParams(stats);
        }
        for(int i = 0; i < later[player].length; i++) {
            if(later[player][i]) {
                stats.visits[i]++;
                stats.totals[i] += outcome;
            }
        }
    }

    // Inspection helpers used by tests. These return raw AMAF counts and means before the prior.
    protected int getRaveVisits(ATPZSQNode node, Action action) {
        if(!(node.getExtraParams() instanceof RaveStats stats)) return 0;
        int dimension = ((GameView)node.getState()).getBoard().getDim();
        return stats.visits[action.coordinate().y() * dimension + action.coordinate().x()];
    }

    protected Double getRaveValue(ATPZSQNode node, Action action) {
        int visits = getRaveVisits(node, action);
        if(visits == 0) return null;
        RaveStats stats = (RaveStats)node.getExtraParams();
        int dimension = ((GameView)node.getState()).getBoard().getDim();
        return stats.totals[action.coordinate().y() * dimension + action.coordinate().x()] / visits;
    }

    // Orchestrate one real decision. Repeat selection, expansion, rollout, and backup.
    // The returned node is a direct child of the root; its incoming action is the real move.
    @Override
    public ATPZSQNode treeSearch(final ATPZSQNode rootNode) {
        if(rootNode.isTerminal()) return rootNode;
        long budget = this.getMaxThiningTimeInMs();
        // Leave 10% of the time allowance (at least 2 ms) for finishing an iteration and
        // returning. The tactical shortcut and the search loop share this cutoff.
        long deadline = System.nanoTime() + Math.max(0, budget - Math.max(2, budget / 10)) * 1000000L;
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
            // The extra-credit agent normally returns the most sampled root move. RAVE affects
            // this indirectly by guiding where the search spends its trials.
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
