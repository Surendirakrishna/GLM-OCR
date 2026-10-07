import { Copy } from 'lucide-react'
import { copyToClipboard } from '@/libs/utils'
import { toast } from 'sonner'

interface CopyButtonProps {
	content: string
	className?: string
}

export function CopyButton({ content, className = '' }: CopyButtonProps) {
	const handleCopy = async () => {
		if (!content) return
		const success = await copyToClipboard(content)
		if (success) {
			toast.success('Copied')
		} else {
			console.error('Copy failed')
		}
	}

	return (
		<button
			data-ocr-copy-button
			onClick={handleCopy}
			className={`absolute py-1 px-3 -top-6 right-0 h-6 flex items-center justify-center gap-1 z-10 backdrop-blur-sm pointer-events-auto bg-black/65 text-white rounded-md cursor-pointer text-nowrap ${className}`}
		>
			<Copy size={14} strokeWidth={1.5} />
			<span>Copy</span>
		</button>
	)
}
