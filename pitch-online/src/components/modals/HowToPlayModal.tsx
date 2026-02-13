import { Button, Modal } from "react-bootstrap";

interface HowToPlayModalProps {
  show: boolean;
  onClose: () => void;
}

export default function HowToPlayModal({ show, onClose }: HowToPlayModalProps) {
  return (
    <Modal show={show} centered scrollable size="lg" onHide={onClose}>
      <Modal.Header closeButton>
        <Modal.Title>How to Play Pitch</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Section title="Overview">
          <p>Pitch is a trick-taking card game for 4 players in 2 teams. Teammates sit across from each other (seats 0 &amp; 2 vs seats 1 &amp; 3). The deck is 52 cards plus 2 jokers. First team to <strong>54 points</strong> wins.</p>
        </Section>

        <Section title="Round Flow">
          <ol className="mb-0">
            <li><strong>Deal</strong> &mdash; Each player receives 9 cards.</li>
            <li><strong>Bidding</strong> &mdash; Starting left of the dealer, players bid (5&ndash;8, Moon, or Double Moon) or pass. The highest bidder wins and chooses the trump suit. The dealer cannot pass if no one else has bid.</li>
            <li><strong>Discard &amp; Fill</strong> &mdash; All non-trump cards are discarded. Each player draws back up to 6 cards.</li>
            <li><strong>Play Tricks</strong> &mdash; The bid winner leads the first trick. Each trick, players play one card. Highest trump wins the trick and leads the next one.</li>
          </ol>
        </Section>

        <Section title="What Can Be Played?">
          <p className="mb-1">Only these cards are valid plays:</p>
          <ul className="mb-0">
            <li><strong>Trump suit</strong> cards</li>
            <li><strong>Jokers</strong> (both are always playable)</li>
            <li><strong>Off-Jack</strong> &mdash; the Jack of the same color as trump (e.g., if trump is Spades, the Jack of Clubs is the off-jack)</li>
          </ul>
        </Section>

        <Section title="Scoring Cards">
          <p className="mb-1">Each round has <strong>10 points</strong> available:</p>
          <table className="table table-sm table-dark table-bordered mb-0" style={{ maxWidth: 300 }}>
            <thead>
              <tr><th>Card</th><th>Points</th></tr>
            </thead>
            <tbody>
              <tr><td>Ace of trump</td><td>1</td></tr>
              <tr><td>Jack of trump</td><td>1</td></tr>
              <tr><td>Off-Jack</td><td>1</td></tr>
              <tr><td>10 of trump</td><td>1</td></tr>
              <tr><td>3 of trump</td><td>3</td></tr>
              <tr><td>2 of trump</td><td>1</td></tr>
              <tr><td>Each Joker</td><td>1 (x2)</td></tr>
            </tbody>
          </table>
        </Section>

        <Section title="End-of-Round Scoring">
          <ul className="mb-0">
            <li><strong>Normal bid (5&ndash;8):</strong> If the bidding team wins at least as many points as they bid, they score all the points they won. If they fall short, they <em>lose</em> their bid amount.</li>
            <li><strong>Non-bidding team:</strong> Always scores whatever points they won, regardless of the bid.</li>
            <li><strong>Moon (all 10 points):</strong> Win all 10 = +20 points. Fall short = &minus;20 points.</li>
            <li><strong>Double Moon:</strong> Win all 10 = +40 points. Fall short = &minus;40 points.</li>
          </ul>
        </Section>

        <Section title="Winning" last>
          <p className="mb-0">The first team to reach <strong>54 points</strong> wins the game.</p>
        </Section>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="outline-light" onClick={onClose}>Got it</Button>
      </Modal.Footer>
    </Modal>
  );
}

function Section({ title, children, last }: { title: string; children: React.ReactNode; last?: boolean }) {
  return (
    <div className={last ? "" : "mb-4"}>
      <h6 className="text-warning">{title}</h6>
      {children}
    </div>
  );
}
