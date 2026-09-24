const mobileTableOfContentsScript = String.raw`
function setupMobileTableOfContents() {
  document.documentElement.classList.remove("mobile-toc-open")

  const root = document.querySelector("[data-mobile-toc]")
  if (!root) return

  const trigger = root.querySelector(".mobile-toc-trigger")
  const backdrop = root.querySelector(".mobile-toc-backdrop")
  const panel = root.querySelector(".mobile-toc-panel")
  const closeButton = root.querySelector(".mobile-toc-close")
  const links = Array.from(root.querySelectorAll(".mobile-toc-panel a"))

  if (!trigger || !backdrop || !panel || !closeButton) return

  let lastFocused = null
  let focusFrame

  const setOpen = (open, restoreFocus = false) => {
    root.classList.toggle("is-open", open)
    trigger.setAttribute("aria-expanded", String(open))
    panel.setAttribute("aria-hidden", String(!open))
    document.documentElement.classList.toggle("mobile-toc-open", open)

    if (focusFrame !== undefined) {
      cancelAnimationFrame(focusFrame)
      focusFrame = undefined
    }

    if (open) {
      lastFocused = document.activeElement instanceof HTMLElement ? document.activeElement : trigger
      const activeLinks = panel.querySelectorAll("a.is-active")
      activeLinks.item(activeLinks.length - 1)?.scrollIntoView({ block: "nearest" })
      focusFrame = requestAnimationFrame(() => closeButton.focus())
    } else if (restoreFocus) {
      ;(lastFocused ?? trigger).focus({ preventScroll: true })
    }
  }

  const open = () => setOpen(true)
  const close = () => setOpen(false, true)
  const closeAfterNavigation = () => setOpen(false)
  const desktopMedia = window.matchMedia("(min-width: 1200px)")
  const closeOnDesktop = (event) => {
    if (event.matches) setOpen(false)
  }

  const onKeyDown = (event) => {
    if (!root.classList.contains("is-open")) return

    if (event.key === "Escape") {
      event.preventDefault()
      close()
      return
    }

    if (event.key !== "Tab") return

    const focusable = Array.from(
      panel.querySelectorAll(
        'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ),
    )
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (!first || !last) return

    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }

  trigger.addEventListener("click", open)
  backdrop.addEventListener("click", close)
  closeButton.addEventListener("click", close)
  links.forEach((link) => link.addEventListener("click", closeAfterNavigation))
  document.addEventListener("keydown", onKeyDown)
  desktopMedia.addEventListener("change", closeOnDesktop)

  window.addCleanup(() => {
    setOpen(false)
    trigger.removeEventListener("click", open)
    backdrop.removeEventListener("click", close)
    closeButton.removeEventListener("click", close)
    links.forEach((link) => link.removeEventListener("click", closeAfterNavigation))
    document.removeEventListener("keydown", onKeyDown)
    desktopMedia.removeEventListener("change", closeOnDesktop)
  })
}

document.addEventListener("nav", setupMobileTableOfContents)
`

export default mobileTableOfContentsScript
