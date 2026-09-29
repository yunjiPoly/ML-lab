/**
 * Card-shaped guide drawn over the live camera preview: a centered portrait
 * rectangle with the 59:86 mm proportions of a Yu-Gi-Oh! card, a translucent
 * border, and a darkened area outside it (done with a huge box-shadow spread).
 */
export default function CardGuideOverlay() {
  return (
    <div className="guide" aria-hidden="true">
      <div className="guide__frame" />
      <p className="guide__hint">Fill the frame with the card, text upright</p>
    </div>
  );
}
