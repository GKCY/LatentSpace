const leftSidebarScript = String.raw`
let leftSidebarCollapsed = false
try {
  leftSidebarCollapsed = localStorage.getItem("left-sidebar-collapsed") === "true"
} catch {}

function setupLeftSidebar() {
  const toggle = document.querySelector("[data-left-sidebar-toggle]")
  const label = toggle?.querySelector("[data-left-sidebar-label]")
  const sidebar = document.getElementById("left-sidebar")
  const mobile = window.matchMedia("(max-width: 800px)")

  const sync = () => {
    const collapsed = Boolean(toggle && sidebar && leftSidebarCollapsed && !mobile.matches)
    document.documentElement.setAttribute("data-left-sidebar-collapsed", String(collapsed))
    if (!toggle || !sidebar) return

    toggle.setAttribute("aria-expanded", String(!collapsed))
    toggle.setAttribute("aria-label", collapsed ? "展开左侧栏" : "收起左侧栏")
    toggle.setAttribute("title", collapsed ? "展开左侧栏" : "收起左侧栏")
    if (label) label.textContent = collapsed ? "展开侧栏" : "收起侧栏"
    sidebar.inert = collapsed
  }

  const onClick = () => {
    leftSidebarCollapsed = !leftSidebarCollapsed
    try {
      localStorage.setItem("left-sidebar-collapsed", String(leftSidebarCollapsed))
    } catch {}
    sync()
  }

  sync()
  if (!toggle || !sidebar) return

  toggle.addEventListener("click", onClick)
  mobile.addEventListener("change", sync)
  window.addCleanup(() => {
    toggle.removeEventListener("click", onClick)
    mobile.removeEventListener("change", sync)
  })
}

document.addEventListener("nav", setupLeftSidebar)
`

export default leftSidebarScript
