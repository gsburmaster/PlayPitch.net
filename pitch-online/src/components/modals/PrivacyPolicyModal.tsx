import { Button, Modal } from "react-bootstrap";

interface PrivacyPolicyModalProps {
  show: boolean;
  onClose: () => void;
}

<<<<<<< HEAD
export default function PrivacyPolicyModal({ show, onClose }: PrivacyPolicyModalProps) {
=======
export default function PrivacyPolicyModal({
  show,
  onClose,
}: PrivacyPolicyModalProps) {
>>>>>>> 50f336509c1d29ee831e1c1e26dbfe2e5994ccc5
  return (
    <Modal show={show} centered scrollable size="lg" onHide={onClose}>
      <Modal.Header closeButton>
        <Modal.Title>Privacy Policy</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Section title="What We Collect">
<<<<<<< HEAD
          <p>When you play, we collect only your <strong>display name</strong> (the nickname you enter on the home screen). This is stored in server memory only and is never saved to a database or written to disk.</p>
        </Section>

        <Section title="No Cookies or Tracking">
          <p>This site does not use cookies, localStorage, analytics, or any third-party tracking scripts.</p>
        </Section>

        <Section title="Data Retention">
          <p>All game data (your display name, game state, and room information) exists only in server memory and is automatically deleted when your game room expires &mdash; at most 30 minutes after the last activity. There is no persistent storage.</p>
        </Section>

        <Section title="AI Players">
          <p>When AI opponents fill empty seats, the AI model processes only game state (cards, bids, scores). No personal information is used for AI inference.</p>
        </Section>

        <Section title="Third Parties">
          <p>This site does not share any data with third parties. All fonts and assets are self-hosted.</p>
        </Section>

        <Section title="Contact">
          <p className="mb-0">Questions? Reach out at <strong>playpitch.net</strong>.</p>
        </Section>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="success" onClick={onClose}>Got it</Button>
=======
          <p>
            When you play, we collect only your <strong>display name</strong>{" "}
            (the nickname you enter on the home screen). This is stored in
            server memory only and is never saved to a database or written to
            disk.
          </p>
        </Section>

        <Section title="No Cookies or Tracking">
          <p>
            This site does not use cookies, localStorage, analytics, or any
            third-party tracking scripts.
          </p>
        </Section>

        <Section title="Data Retention">
          <p>
            All game data (your display name, game state, and room information)
            exists only in server memory and is automatically deleted when your
            game room expires &mdash; at most 30 minutes after the last
            activity. There is no persistent storage.
          </p>
        </Section>

        <Section title="AI Players">
          <p>
            When AI opponents fill empty seats, the AI model processes only game
            state (cards, bids, scores). No personal information is used for AI
            inference.
          </p>
        </Section>

        <Section title="Third Parties">
          <p>
            This site does not share any data with third parties. All fonts and
            assets are self-hosted.
          </p>
        </Section>

        {/* <Section title="Contact">
          <p className="mb-0">Questions? Reach out at <strong>playpitch.net</strong>.</p>
        </Section> */}
      </Modal.Body>
      <Modal.Footer>
        <Button variant="success" onClick={onClose}>
          Got it
        </Button>
>>>>>>> 50f336509c1d29ee831e1c1e26dbfe2e5994ccc5
      </Modal.Footer>
    </Modal>
  );
}

<<<<<<< HEAD
function Section({ title, children }: { title: string; children: React.ReactNode }) {
=======
function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
>>>>>>> 50f336509c1d29ee831e1c1e26dbfe2e5994ccc5
  return (
    <div className="rules-section">
      <h6 className="rules-section-title">{title}</h6>
      {children}
    </div>
  );
}
