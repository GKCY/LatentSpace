import { i18n } from "../i18n"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

import mobileTocScript from "./scripts/mobileTableOfContents"

type TocEntry = {
  depth: number
  text: string
  slug: string
}

export default (() => {
  const MobileTableOfContents: QuartzComponent = ({ fileData, cfg }: QuartzComponentProps) => {
    const toc = (fileData as typeof fileData & { toc?: TocEntry[] }).toc
    if (!Array.isArray(toc) || toc.length === 0) return null

    const title = i18n(cfg.locale).components.tableOfContents.title

    return (
      <div class="mobile-toc" data-mobile-toc>
        <button
          type="button"
          class="mobile-toc-trigger"
          aria-label={`打开${title}`}
          aria-controls="mobile-toc-panel"
          aria-expanded="false"
        >
          <svg
            aria-hidden="true"
            xmlns="http://www.w3.org/2000/svg"
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
          >
            <line x1="8" y1="6" x2="21" y2="6" />
            <line x1="8" y1="12" x2="21" y2="12" />
            <line x1="8" y1="18" x2="21" y2="18" />
            <line x1="3" y1="6" x2="3.01" y2="6" />
            <line x1="3" y1="12" x2="3.01" y2="12" />
            <line x1="3" y1="18" x2="3.01" y2="18" />
          </svg>
          <span>{title}</span>
        </button>

        <button
          type="button"
          class="mobile-toc-backdrop"
          aria-label={`关闭${title}`}
          aria-hidden="true"
          tabIndex={-1}
        />

        <section
          id="mobile-toc-panel"
          class="mobile-toc-panel"
          role="dialog"
          aria-modal="true"
          aria-hidden="true"
          aria-labelledby="mobile-toc-title"
        >
          <header class="mobile-toc-panel-header">
            <h2 id="mobile-toc-title">{title}</h2>
            <button type="button" class="mobile-toc-close" aria-label={`关闭${title}`}>
              <svg
                aria-hidden="true"
                xmlns="http://www.w3.org/2000/svg"
                width="22"
                height="22"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </header>

          <nav aria-label={title} class="mobile-toc-panel-content">
            <ul>
              {toc.map((entry) => {
                const depth = Math.max(0, Math.min(6, Number(entry.depth) || 0))
                return (
                  <li class={`depth-${depth}`} key={entry.slug}>
                    <a href={`#${entry.slug}`} data-for={entry.slug}>
                      {entry.text}
                    </a>
                  </li>
                )
              })}
            </ul>
          </nav>
        </section>
      </div>
    )
  }

  MobileTableOfContents.afterDOMLoaded = mobileTocScript

  return MobileTableOfContents
}) satisfies QuartzComponentConstructor
