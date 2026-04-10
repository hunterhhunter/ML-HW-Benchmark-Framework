interface ConfirmDialogProps {
  title: string
  message: string
  subMessage?: string
  onConfirm: () => void
  onCancel: () => void
  confirmLabel?: string
}

export default function ConfirmDialog({
  title,
  message,
  subMessage,
  onConfirm,
  onCancel,
  confirmLabel = 'Delete',
}: ConfirmDialogProps) {
  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal confirm-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 className="section-title">{title}</h2>
          <button className="modal-close" onClick={onCancel}>&times;</button>
        </div>
        <p className="confirm-text">{message}</p>
        {subMessage && <p className="confirm-sub">{subMessage}</p>}
        <div className="modal-actions">
          <button className="btn" onClick={onCancel}>Cancel</button>
          <button className="btn btn-danger" onClick={onConfirm}>{confirmLabel}</button>
        </div>
      </div>
    </div>
  )
}
