const rust = import('./pkg');

// list of examples to run
let example_select = document.createElement('select');
example_select.add(document.createElement('option'));
example_select.style.position = 'absolute';
example_select.style.top = 0;
example_select.style.left = 0;
document.body.appendChild(example_select);

// user input, if needed
let user_input_args = document.createElement('input');
user_input_args.style.position = 'absolute';
user_input_args.style.top = 0;
user_input_args.style.right = 0;
user_input_args.hidden = true;
document.body.appendChild(user_input_args);

let canvas = document.createElement('canvas');
canvas.tabIndex = 0;
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
canvas.id = 'luminance-canvas';
canvas.hidden = true;
document.body.appendChild(canvas);

rust
  .then(wasm => {
    // get the showcase
    const showcase = wasm.get_showcase('luminance-canvas');

    // build the <select> shit madness
    const example_names = wasm.examples_names();
    example_names.forEach(name => {
      let option = document.createElement('option');
      option.text = name;
      example_select.add(option);
    });

    example_select.onchange = change => {
      showcase.reset();
      canvas.hidden = change.target.value === '';
    };

    // handle user input
    user_input_args.onchange = change => {
    };

    // transform events into input actions
    canvas.addEventListener('keyup', (event) => {
      switch (event.code) {
        case 'Space':
          if (event.shiftKey) {
            showcase.enqueue_auxiliary_toggle_action();
          } else {
            showcase.enqueue_main_toggle_action();
          }
          break;

        case 'Escape':
          showcase.enqueue_quit_action();
          break;

        case 'KeyA':
          showcase.enqueue_left_action();
          break;

        case 'KeyD':
          showcase.enqueue_right_action();
          break;

        case 'KeyW':
          showcase.enqueue_up_action();
          break;

        case 'KeyS':
          showcase.enqueue_down_action();
          break;

        default:
      }
    });

    window.onresize = () => {
      if (window.innerWidth !== undefined && window.innerHeight !== undefined) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        showcase.enqueue_resized_action(window.width, window.height);
      }
    };

    const renderFrame = (now) => {
      if (example_select.value !== '') {
        const feedback = showcase.render_example(example_select.value, now * 1e-3);

        if (!feedback) {
          example_select.value = '';
          showcase.reset();
          canvas.hidden = true;
        }
      }

      window.requestAnimationFrame(renderFrame);
    };

    window.requestAnimationFrame(renderFrame);
  })
  .catch(console.error);
