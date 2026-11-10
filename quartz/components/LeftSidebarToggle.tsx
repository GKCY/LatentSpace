import { QuartzComponent, QuartzComponentConstructor } from "./types"
import leftSidebarScript from "./scripts/leftSidebar"

export default (() => {
  const LeftSidebarToggle: QuartzComponent = () => (
    <button
      type="button"
      class="left-sidebar-toggle"
      data-left-sidebar-toggle
      aria-controls="left-sidebar"
      aria-expanded="true"
      aria-label="收起左侧栏"
      title="收起左侧栏"
    >
      <svg
        aria-hidden="true"
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.8"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <path d="M9 3v18" />
        <path class="left-sidebar-toggle-arrow" d="m16 9-3 3 3 3" />
      </svg>
      <span data-left-sidebar-label>收起侧栏</span>
    </button>
  )

  LeftSidebarToggle.afterDOMLoaded = leftSidebarScript
  return LeftSidebarToggle
}) satisfies QuartzComponentConstructor
